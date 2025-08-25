/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "warp.h"
#include "cuda_util.h"
#include "bvh.h"
#include "sort.h"

#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define THRUST_IGNORE_CUB_VERSION_CHECK
#define REORDER_HOST_TREE

#include <cub/cub.cuh>


namespace wp
{
    void bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int constructor_type, BVH& bvh);
    void bvh_destroy_host(BVH& bvh);

// for LBVH: this will start with some muted leaf nodes, but that is okay, we can still trace up because there parents information is still valid
// the only thing worth mentioning is that when the parent leaf node is also a leaf node, we need to recompute its bounds, since their child information are lost
// for a compact tree such as those from SAH or Median constructor, there is no muted leaf nodes
__global__ void bvh_refit_kernel(int n, const int* __restrict__ parents, int* __restrict__ child_count, const int* __restrict__ primitive_indices, BVHPackedNodeHalf* __restrict__ node_lowers, BVHPackedNodeHalf* __restrict__ node_uppers, const vec3* __restrict__ item_lowers, const vec3* __restrict__ item_uppers)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        bool leaf = node_lowers[index].b;
        int parent = parents[index];

        if (leaf)
        {
            BVHPackedNodeHalf& lower = node_lowers[index];
            BVHPackedNodeHalf& upper = node_uppers[index];
            // update the leaf node

            // only need to compute bound when this is a valid leaf node
            if (!node_lowers[parent].b)
            {
                const int start = lower.i;
                const int end = upper.i;

                bounds3 bound;
                for (int primitive_counter = start; primitive_counter < end; primitive_counter++)
                {
                    const int primitive = primitive_indices[primitive_counter];
                    bound.add_bounds(item_lowers[primitive], item_uppers[primitive]);
                }
                (vec3&)lower = bound.lower;
                (vec3&)upper = bound.upper;
            }
        }
        else
        {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;)
        {
            parent = parents[index];
            // reached root
            if (parent == -1)
                return;

            // ensure all writes are visible
            __threadfence();
         
            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the next parent in the hierarchy
            if (finished == 1)
            {
                BVHPackedNodeHalf& parent_lower = node_lowers[parent];
                BVHPackedNodeHalf& parent_upper = node_uppers[parent];
                if (parent_lower.b)
                    // a packed leaf node can still be a parent in LBVH, we need to recompute its bounds
                    // since we've lost its left and right child node index in the muting process
                {
                    // update the leaf node
                    int parent_parent = parents[parent];;

                    // only need to compute bound when this is a valid leaf node
                    if (!node_lowers[parent_parent].b)
                    {
                        const int start = parent_lower.i;
                        const int end = parent_upper.i;
                        bounds3 bound;
                        for (int primitive_counter = start; primitive_counter < end; primitive_counter++)
                        {
                            const int primitive = primitive_indices[primitive_counter];
                            bound.add_bounds(item_lowers[primitive], item_uppers[primitive]);
                        }

                        (vec3&)parent_lower = bound.lower;
                        (vec3&)parent_upper = bound.upper;
                    }
                }
                else
                {
                    const int left_child = parent_lower.i;
                    const int right_child = parent_upper.i;

                    vec3 left_lower = (vec3&)(node_lowers[left_child]);
                    vec3 left_upper = (vec3&)(node_uppers[left_child]);
                    vec3 right_lower = (vec3&)(node_lowers[right_child]);
                    vec3 right_upper = (vec3&)(node_uppers[right_child]);

                    // union of child bounds
                    vec3 lower = min(left_lower, right_lower);
                    vec3 upper = max(left_upper, right_upper);

                    // write new BVH nodes
                    (vec3&)parent_lower = lower;
                    (vec3&)parent_upper = upper;
                }
                // move onto processing the parent
                index = parent;
            }
            else
            {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }		
    }
}


void bvh_refit_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);

    // clear child counters
    wp_memset_device(WP_CURRENT_CONTEXT, bvh.node_counts, 0, sizeof(int) * bvh.max_nodes);
    wp_launch_device(WP_CURRENT_CONTEXT, bvh_refit_kernel, bvh.num_leaf_nodes, (bvh.num_leaf_nodes, bvh.node_parents, bvh.node_counts, bvh.primitive_indices, bvh.node_lowers, bvh.node_uppers, bvh.item_lowers, bvh.item_uppers));
}


/////////////////////////////////////////////////////////////////////////////////////////////

// Create a linear BVH as described in Fast and Simple Agglomerative LBVH construction
// this is a bottom-up clustering method that outputs one node per-leaf 
//
class LinearBVHBuilderGPU
{
public:

    LinearBVHBuilderGPU();
    ~LinearBVHBuilderGPU();

    // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for each triangle
    void build(BVH& bvh, const vec3* item_lowers, const vec3* item_uppers, int num_items, bounds3* total_bounds);

private:

    // temporary data used during building
    int* indices;
    int* keys;
    int* deltas;
    int* range_lefts;
    int* range_rights;
    int* num_children;

    // bounds data when total item bounds built on GPU
    vec3* total_lower;
    vec3* total_upper;
    vec3* total_inv_edges;
};

////////////////////////////////////////////////////////



__global__ void compute_morton_codes(const vec3* __restrict__ item_lowers, const vec3* __restrict__ item_uppers, int n, const vec3* grid_lower, const vec3* grid_inv_edges, int* __restrict__ indices, int* __restrict__ keys)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        vec3 lower = item_lowers[index];
        vec3 upper = item_uppers[index];

        vec3 center = 0.5f*(lower+upper);

        vec3 local = cw_mul((center-grid_lower[0]), grid_inv_edges[0]);
        
        // 10-bit Morton codes stored in lower 30bits (1024^3 effective resolution)
        int key = morton3<1024>(local[0], local[1], local[2]);

        indices[index] = index;
        keys[index] = key;
    }
}

// calculate the index of the first differing bit between two adjacent Morton keys
__global__ void compute_key_deltas(const int* __restrict__ keys, int* __restrict__ deltas, int n)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        int a = keys[index];
        int b = keys[index+1];

        int x = a^b;

        deltas[index] = x;// __clz(x);
    }
}

__global__ void build_leaves(const vec3* __restrict__ item_lowers, const vec3* __restrict__ item_uppers, int n, const int* __restrict__ indices, int* __restrict__ range_lefts, int* __restrict__ range_rights, BVHPackedNodeHalf* __restrict__ lowers, BVHPackedNodeHalf* __restrict__ uppers)
{
    const int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        const int item = indices[index];

        vec3 lower = item_lowers[item];
        vec3 upper = item_uppers[item];

        // write leaf nodes 
        lowers[index] = make_node(lower, item, true);
        uppers[index] = make_node(upper, item, false);

        // write leaf key ranges
        range_lefts[index] = index;
        range_rights[index] = index;
    }
}

// this bottom-up process assigns left and right children and combines bounds to form internal nodes
// there is one thread launched per-leaf node, each thread calculates it's parent node and assigns
// itself to either the left or right parent slot, the last child to complete the parent and moves
// up the hierarchy
__global__ void build_hierarchy(int n, int* root, const int* __restrict__ deltas,  int* __restrict__ num_children, const int* __restrict__ primitive_indices, volatile int* __restrict__ range_lefts, volatile int* __restrict__ range_rights, volatile int* __restrict__ parents, volatile BVHPackedNodeHalf* __restrict__ lowers, volatile BVHPackedNodeHalf* __restrict__ uppers)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;

    if (index < n)
    {
        const int internal_offset = n;

        for (;;)
        {
            int left = range_lefts[index];
            int right = range_rights[index];

            // check if we are the root node, if so then store out our index and terminate
            if (left == 0 && right == n-1)
            {					
                *root = index;
                parents[index] = -1;

                break;
            }

            int childCount = 0;

            int parent;

            bool parent_right = false;
            if (left == 0) 
            {
                parent_right = true;
            }
            else if ((right != n - 1 && deltas[right] <= deltas[left - 1]))
            {
                // tie breaking, this avoid always choosing the right node which can result in a very deep tree
                // generate a pseudo-random binary value to randomly choose left or right groupings
                // since the primitives with same Morton code are not sorted at all, determining order based on primitive_indices may also be unreliable.  
                // Here, the decision is made using the XOR result of whether the keys before and after the internal node are divisible by 2.  
                if (deltas[right] == deltas[left - 1])
                {
                    parent_right = (primitive_indices[left - 1] % 2) ^ (primitive_indices[right] % 2);
                }
                else
                {
                    parent_right = true;
                }
            }

            if (parent_right)
            {
                parent = right + internal_offset;

                // set parent left child
                parents[index] = parent;
                lowers[parent].i = index;
                range_lefts[parent] = left;

                // ensure above writes are visible to all threads
                __threadfence();
                
                childCount = atomicAdd(&num_children[parent], 1);
            }
            else
            {
                parent = left + internal_offset - 1;
                
                // set parent right child
                parents[index] = parent;
                uppers[parent].i = index;
                range_rights[parent] = right;

                // ensure above writes are visible to all threads
                __threadfence();
                
                childCount = atomicAdd(&num_children[parent], 1);
            }

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the next parent in the hierarchy
            if (childCount == 1)
            {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 left_lower = vec3(lowers[left_child].x,
                                       lowers[left_child].y, 
                                       lowers[left_child].z);

                vec3 left_upper = vec3(uppers[left_child].x,
                                       uppers[left_child].y, 
                                       uppers[left_child].z);

                vec3 right_lower = vec3(lowers[right_child].x,
                                        lowers[right_child].y,
                                        lowers[right_child].z);


                vec3 right_upper = vec3(uppers[right_child].x, 
                                        uppers[right_child].y, 
                                        uppers[right_child].z);

                // bounds_union of child bounds
                vec3 lower = min(left_lower, right_lower);
                vec3 upper = max(left_upper, right_upper);
                
                // write new BVH nodes
                make_node(lowers+parent, lower, left_child, false);
                make_node(uppers+parent, upper, right_child, false);

                // move onto processing the parent
                index = parent;
            }
            else
            {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }		
    }
}

/*
* LBVH uses a bottom-up constructor which makes variable-sized leaf nodes more challenging to achieve. 
* Simply splitting the ordered primitives into uniform groups of size BVH_LEAF_SIZE will result in poor
* quality. Instead, after the hierarchy is built, we convert any intermediate node whose size is 
* <= BVH_LEAF_SIZE into a new leaf node. This process is done using the new kernel function called 
* mark_packed_leaf_nodes .
*/
__global__ void mark_packed_leaf_nodes(int n, const int* __restrict__ range_lefts, const int* __restrict__ range_rights, const int* __restrict__ parents,
    BVHPackedNodeHalf* __restrict__ lowers, BVHPackedNodeHalf* __restrict__ uppers)
{
    int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index < n)
    {
        // mark the node as leaf if its range is less than LEAF_SIZE_LBVH or it is deeper than BVH_QUERY_STACK_SIZE
        // this will forever mute its child nodes so that they will never be accessed

        // calculate depth
        int depth = 1;
        int parent = parents[node_index];
        while (parent != -1)
        {
            int old_parent = parent;
            parent = parents[parent];
            depth++;
        }

        int left = range_lefts[node_index];
        // the LBVH constructor's range is defined as left <= i <= right
        // we need to convert it to our convention: left <= i < right
        int right = range_rights[node_index] + 1;
        if (right - left <= BVH_LEAF_SIZE || depth >= BVH_QUERY_STACK_SIZE)
        {
            lowers[node_index].b = 1;
            lowers[node_index].i = left;
            uppers[node_index].i = right;
        }
    }
}


CUDA_CALLABLE inline vec3 Vec3Max(const vec3& a, const vec3& b) { return wp::max(a, b); }
CUDA_CALLABLE inline vec3 Vec3Min(const vec3& a, const vec3& b) { return wp::min(a, b); }

__global__ void compute_total_bounds(const vec3* item_lowers, const vec3* item_uppers, vec3* total_lower, vec3* total_upper, int num_items)
{
     typedef cub::BlockReduce<vec3, 256> BlockReduce;

     __shared__ typename BlockReduce::TempStorage temp_storage;

     const int blockStart = blockDim.x*blockIdx.x;
     const int numValid = ::min(num_items-blockStart, blockDim.x);

     const int tid = blockStart + threadIdx.x;

     if (tid < num_items)
     {
        vec3 lower = item_lowers[tid];
        vec3 upper = item_uppers[tid];

         vec3 block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max, numValid);

         // sync threads because second reduce uses same temp storage as first
         __syncthreads();

         vec3 block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min, numValid);

         if (threadIdx.x == 0)
         {
             // write out block results, expanded by the radius
             atomic_max(total_upper, block_upper);
             atomic_min(total_lower, block_lower);
         }	 
    }
}

// compute inverse edge length, this is just done on the GPU to avoid a CPU->GPU sync point
__global__ void compute_total_inv_edges(const vec3* total_lower, const vec3* total_upper, vec3* total_inv_edges)
{
    vec3 edges = (total_upper[0]-total_lower[0]);
    edges += vec3(0.0001f);

    total_inv_edges[0] = vec3(1.0f/edges[0], 1.0f/edges[1], 1.0f/edges[2]);
}



LinearBVHBuilderGPU::LinearBVHBuilderGPU() 
    : indices(NULL)
    , keys(NULL)
    , deltas(NULL)
    , range_lefts(NULL)
    , range_rights(NULL)
    , num_children(NULL)
    , total_lower(NULL)
    , total_upper(NULL)
    , total_inv_edges(NULL)
{
    total_lower = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3));
    total_upper = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3));
    total_inv_edges = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3));
}

LinearBVHBuilderGPU::~LinearBVHBuilderGPU()
{
    wp_free_device(WP_CURRENT_CONTEXT, total_lower);
    wp_free_device(WP_CURRENT_CONTEXT, total_upper);
    wp_free_device(WP_CURRENT_CONTEXT, total_inv_edges);
}


static void print_verts(FILE* f, BVHPackedNodeHalf const& lower, BVHPackedNodeHalf const& upper) 
{
    auto getCorner = [&lower, &upper](int corner) -> vec3 {
        bool bits[3] = { (corner & 1)!=0, (corner & 2) != 0, (corner & 4) != 0 };      
        return { bits[0] ? lower.x : upper.x, bits[1] ? lower.y : upper.y, bits[2] ? lower.z : upper.z };
    };

    for( uint8_t i = 0; i < 8; ++i )
    {
        vec3 corner = getCorner(i);
        fprintf( f, "v %f %f %f\n", corner[0], corner[1], corner[2] );
    }
}

static void print_face( FILE* f, uint32_t ofs ) 
{
    fprintf( f, "f %d %d %d %d\n", ofs+1, ofs+5, ofs+6, ofs+2 );
    fprintf( f, "f %d %d %d %d\n", ofs+3, ofs+7, ofs+8, ofs+4 );
    fprintf( f, "f %d %d %d %d\n", ofs+1, ofs+3, ofs+7, ofs+5 );
    fprintf( f, "f %d %d %d %d\n", ofs+5, ofs+7, ofs+8, ofs+6 );
    fprintf( f, "f %d %d %d %d\n", ofs+6, ofs+8, ofs+4, ofs+2 );
    fprintf( f, "f %d %d %d %d\n", ofs+2, ofs+4, ofs+3, ofs+1 );
};

static void debug(BVH const& bvh)
{
    wp_cuda_context_synchronize(WP_CURRENT_CONTEXT);

    int root = -1;
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, &root, bvh.root, sizeof(int));

    wp_cuda_context_synchronize(WP_CURRENT_CONTEXT);


    std::vector<BVHPackedNodeHalf> lowers(bvh.max_nodes), uppers(bvh.max_nodes);
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, lowers.data(), bvh.node_lowers, bvh.max_nodes * sizeof(BVHPackedNodeHalf));
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, uppers.data(), bvh.node_uppers, bvh.max_nodes * sizeof(BVHPackedNodeHalf));

    static uint32_t fcount = 0;
    char filepath[256];
    snprintf(filepath, std::size(filepath), "D:/tmp/debug_warp_bvh.%04d.obj", fcount++);
    FILE* f = fopen(filepath, "w");

    uint32_t vert_count = 0;
#if 1
    uint32_t max_depth = 0;
    auto traverse = [f, &lowers, &uppers, &max_depth](const auto& self, uint32_t node_id, uint32_t depth, uint32_t& vert_count) -> void {
       
        max_depth = max(max_depth, depth);

        BVHPackedNodeHalf const& lower = lowers[node_id];
        BVHPackedNodeHalf const& upper = uppers[node_id];

        fprintf(f, "g node_%04d_l_%04d_r_%04d_d_%d%s\n", 
            node_id, lower.i, upper.i, depth, lower.b ? "_leaf" : "");

        print_verts(f, lower, upper);
        print_face( f, vert_count );
        vert_count += 8;

        if( !lower.b )
        {
            self( self, lower.i, depth + 1, vert_count );
            self( self, upper.i, depth + 1, vert_count );
        }
    };

    traverse( traverse, root, 1, vert_count );
    printf("\nXXXX debug : %s root=%d, max_depth=%d max_nodes=%d\n", filepath, root, max_depth, bvh.max_nodes);    
#else    
    for( uint32_t i = 0; i < (uint32_t)nodes.size(); ++i )
    {
        node_type const& n = nodes[i];        
        if( n.leaf )
        {
            fprintf( f, "g node_%04d_l_%04d_r_%04d_%s\n", 
                i, n.left_child, n.right_child, n.leaf ? "_leaf" : "" );

            print_verts( f, n.aabb );
            print_face( f, vert_count );

            vert_count += 8;
        }
    }
#endif
    fclose( f );
}

static bounds3 _total_bounds;
static vec3 _total_inv_edges;

class StopwatchGPU
{
public:
    void StopwatchGPU::start( CUstream stream )
    {
        // all but 'stopped' states are valid, so can advance up to 3 times
        assert( state != State::ticking );
        cudaEventRecord( *m_startEvent, m_stream = stream );
        state = State::ticking;
    }

    void StopwatchGPU::stop()
    {
        assert( state == State::ticking );
        cudaEventRecord( *m_stopEvent, m_stream );
        state = State::stopped;
    }

    void StopwatchGPU::sync()
    {
        assert( state == State::stopped );
        cudaEventSynchronize( *m_stopEvent );
        state = State::synced;
    }

    std::optional<float> StopwatchGPU::elapsed()
    {
        if( state == State::reset )
            return {};

        assert( state == State::stopped );

        sync();

        state         = State::reset;
        float elapsed = 0.f;
        cudaEventElapsedTime( &elapsed, *m_startEvent, *m_stopEvent );
        return elapsed;
    }

    std::optional<float> StopwatchGPU::elapsedAsync()
    {
        if( state == State::reset )
            return {};

        // user is responsible for device sync, so we can't track it
        assert( state != State::ticking );

        state         = State::reset;
        float elapsed = 0.f;
        cudaEventElapsedTime( &elapsed, *m_startEvent, *m_stopEvent );
        return elapsed;
    }

  private:

    struct Event
    {
        CUevent m_event = nullptr;
        Event::Event()
        {
            cudaEventCreate( &m_event );
        }

        Event::~Event() noexcept
        {
            if( m_event )
                cudaEventDestroy( m_event );
        }

        Event( Event&& e )
        {
            m_event   = e.m_event;
            e.m_event = nullptr;
        }
        Event&  operator=( const Event& event ) = delete;
        CUevent operator*() { return m_event; }
    };

    Event m_startEvent;
    Event m_stopEvent;

    CUstream m_stream = nullptr;

    enum class State : uint8_t
    {
        reset = 0,
        ticking,
        stopped,
        synced
    } state = State::reset;
};

struct Profiler 
{
    enum class Timer : uint8_t {
        extents = 0,
        inv_extents,
        morton_codes,
        sort_pairs,
        key_deltas,
        build_leaves,
        build_hierarchy,
        pack_leaves,
        global,
        count
    };

    static constexpr char const* timer_names[] = { 
        "compute extents", 
        "compute inverse extents total" ,
        "compute morton codes",
        "sort pairs",
        "compute key deltas",
        "build leaves",
        "build hierarchy",
        "mark packed leaf nodes",
        "global"
    };

    static_assert( (uint8_t)Timer::count == std::size(timer_names) );

    std::array<StopwatchGPU, (uint8_t)Timer::count> timers;

    StopwatchGPU& operator [](Timer i) { return timers[(uint8_t)i]; }

    void print_timers(FILE* f = stdout)
    {
        cudaDeviceSynchronize();
        float total = 0.f;
        for( uint8_t i = 0; i < (uint8_t)Timer::global; ++i )
        {
            StopwatchGPU& timer = timers[i];
            if( auto elapsed = timer.elapsedAsync() )
            {
                printf(" - %s : %.2f ms\n", timer_names[i], *elapsed);
                total += *elapsed;
            }
        }
        printf( " - total : %.2f ms\n", total );
        
        if( auto elapsed = timers[(uint8_t)Timer::global].elapsedAsync() )
            printf( " - %s : %.2f ms\n", timer_names[(uint8_t)Timer::global], *elapsed );
    }
} profiler;

void LinearBVHBuilderGPU::build(BVH& bvh, const vec3* item_lowers, const vec3* item_uppers, int num_items, bounds3* total_bounds)
{
    StopwatchGPU timer;

    // allocate temporary memory used during  building
    indices = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*num_items*2); 	// *2 for radix sort
    keys = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*num_items*2);	    // *2 for radix sort
    deltas = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*num_items);    	// highest differentiating bit between keys for item i and i+1
    range_lefts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*bvh.max_nodes);
    range_rights = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*bvh.max_nodes);
    num_children = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int)*bvh.max_nodes);

    //if (!_total_bounds.empty())
    //    total_bounds = &_total_bounds;

     
    profiler[Profiler::Timer::global].start((CUstream)WP_CURRENT_CONTEXT);

    // if total bounds supplied by the host then we just 
    // compute our edge length and upload it to the GPU directly
    if (total_bounds)
    {
        // calculate Morton codes
        vec3 edges = (*total_bounds).edges();
        edges += vec3(0.0001f);

        vec3 inv_edges = _total_bounds.empty() ? vec3(1.0f/edges[0], 1.0f/edges[1], 1.0f/edges[2]) : _total_inv_edges;

        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_lower, &total_bounds->lower[0], sizeof(vec3));
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_upper, &total_bounds->upper[0], sizeof(vec3));
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_inv_edges, &inv_edges[0], sizeof(vec3));
    }
    else
    {
        static vec3 upper(-FLT_MAX);
        static vec3 lower(FLT_MAX);

        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_lower, &lower, sizeof(lower));
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_upper, &upper, sizeof(upper));

        // compute the total bounds on the GPU
        profiler[Profiler::Timer::extents].start((CUstream)WP_CURRENT_CONTEXT);
        wp_launch_device(WP_CURRENT_CONTEXT, compute_total_bounds, num_items, (item_lowers, item_uppers, total_lower, total_upper, num_items));
        profiler[Profiler::Timer::extents].stop();

        // compute the total edge length
        profiler[Profiler::Timer::inv_extents].start((CUstream)WP_CURRENT_CONTEXT);
        wp_launch_device(WP_CURRENT_CONTEXT, compute_total_inv_edges, 1, (total_lower, total_upper, total_inv_edges));
        profiler[Profiler::Timer::inv_extents].stop();

    }

    // assign 30-bit Morton code based on the centroid of each triangle and bounds for each leaf
    profiler[Profiler::Timer::morton_codes].start((CUstream)WP_CURRENT_CONTEXT);
    wp_launch_device(WP_CURRENT_CONTEXT, compute_morton_codes, num_items, (item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys));
    profiler[Profiler::Timer::morton_codes].stop();

    // sort items based on Morton key (note the 32-bit sort key corresponds to the template parameter to morton3, i.e. 3x9 bit keys combined)

    profiler[Profiler::Timer::sort_pairs].start((CUstream)WP_CURRENT_CONTEXT);
    radix_sort_pairs_device(WP_CURRENT_CONTEXT, keys, indices, num_items);
    wp_memcpy_d2d(WP_CURRENT_CONTEXT, bvh.primitive_indices, indices, sizeof(int) * num_items);
    profiler[Profiler::Timer::sort_pairs].stop();

#if 0
    {
        wp_cuda_context_synchronize(WP_CURRENT_CONTEXT);
        vec3 upper(-FLT_MAX), lower(FLT_MAX), inv(FLT_MAX);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, &lower, total_lower, sizeof(vec3));            
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, &upper, total_upper, sizeof(vec3));
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, &inv, total_inv_edges, sizeof(vec3));        
        wp_cuda_context_synchronize(WP_CURRENT_CONTEXT);
        FILE* f = fopen("D:/Work/dev/optix/optix_cluster_bench.clean/_build/warp_log", "w");
        fprintf(f, "AABB (%x %x %x - %x %x %x) inv=(%x %x %x)\n",
                *(uint32_t const*)&lower[0],
                *(uint32_t const*)&lower[1], 
                *(uint32_t const*)&lower[2], 
                *(uint32_t const*)&upper[0], 
                *(uint32_t const*)&upper[1], 
                *(uint32_t const*)&upper[2], 
                *(uint32_t const*)&inv[0], 
                *(uint32_t const*)&inv[1], 
                *(uint32_t const*)&inv[2]);

        std::vector<int> _keys(num_items * 2), _indices(num_items * 2);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _keys.data(), keys, num_items * 2 * sizeof(int));
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _indices.data(), indices, num_items * 2 * sizeof(int));
        for (uint32_t i = 0; i < (num_items * 2); ++i)
            printf(f, "key: %d value: %d\n", _keys[i], _indices[i]);
        fclose(f);
    }
#endif    

    // calculate deltas between adjacent keys
    profiler[Profiler::Timer::key_deltas].start((CUstream)WP_CURRENT_CONTEXT);
    wp_launch_device(WP_CURRENT_CONTEXT, compute_key_deltas, num_items, (keys, deltas, num_items-1));
    profiler[Profiler::Timer::key_deltas].stop();

#if 0
    {
        wp_cuda_context_synchronize(WP_CURRENT_CONTEXT);        
        std::vector<int> _keys(num_items);
        std::vector<int> _deltas(num_items);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _keys.data(), keys, num_items * sizeof(int));
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _deltas.data(), deltas, num_items * sizeof(int));
        FILE* f = fopen("D:/Work/dev/optix/optix_cluster_bench.clean/_build/warp_log", "w");
        for (uint32_t i = 0; i < (num_items); ++i)
            fprintf(f, "key: %d delta: %d\n", _keys[i], _deltas[i]);
        fclose(f);
    }
#endif    

    // initialize leaf nodes
    profiler[Profiler::Timer::build_leaves].start((CUstream)WP_CURRENT_CONTEXT);
    wp_launch_device(WP_CURRENT_CONTEXT, build_leaves, num_items, (item_lowers, item_uppers, num_items, indices, range_lefts, range_rights, bvh.node_lowers, bvh.node_uppers));
    profiler[Profiler::Timer::build_leaves].stop();

#if 0
    {
        wp_cuda_context_synchronize(WP_CURRENT_CONTEXT);
        std::vector<wp::BVHPackedNodeHalf> _node_lowers(bvh.max_nodes), _node_uppers(bvh.max_nodes);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _node_lowers.data(), bvh.node_lowers, bvh.max_nodes * sizeof(wp::BVHPackedNodeHalf));
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _node_uppers.data(), bvh.node_uppers, bvh.max_nodes * sizeof(wp::BVHPackedNodeHalf));

        std::vector<int> _range_lefts(bvh.max_nodes), _range_rights(bvh.max_nodes);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _range_lefts.data(), range_lefts, bvh.max_nodes * sizeof(int));
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _range_rights.data(), range_rights, bvh.max_nodes * sizeof(int));

        FILE* f = fopen("D:/Work/dev/optix/optix_cluster_bench.clean/_build/warp_log", "w");
        for (uint32_t i = 0; i < (bvh.max_nodes); ++i)
        {
            wp::BVHPackedNodeHalf const& lo = _node_lowers[i], up = _node_uppers[i];
            fprintf(f, "(%f %f %f) (%f %f %f) l=%d r=%d b=%d range(%d %d)\n", lo.x, lo.y, lo.z, up.x, up.y, up.z, lo.i, up.i, lo.b, _range_lefts[i], _range_rights[i]);
        }
        fclose(f);
    }
#endif    

    // reset children count, this is our atomic counter so we know when an internal node is complete, only used during building
    wp_memset_device(WP_CURRENT_CONTEXT, num_children, 0, sizeof(int)*bvh.max_nodes);

    // build the tree and internal node bounds
    profiler[Profiler::Timer::build_hierarchy].start((CUstream)WP_CURRENT_CONTEXT);
    wp_launch_device(WP_CURRENT_CONTEXT, build_hierarchy, num_items, (num_items, bvh.root, deltas, num_children, bvh.primitive_indices, range_lefts, range_rights, bvh.node_parents, bvh.node_lowers, bvh.node_uppers));
    profiler[Profiler::Timer::build_hierarchy].stop();

#if 0
    {
        wp_cuda_context_synchronize(WP_CURRENT_CONTEXT);
        std::vector<wp::BVHPackedNodeHalf> _node_lowers(bvh.max_nodes), _node_uppers(bvh.max_nodes);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _node_lowers.data(), bvh.node_lowers, bvh.max_nodes * sizeof(wp::BVHPackedNodeHalf));
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _node_uppers.data(), bvh.node_uppers, bvh.max_nodes * sizeof(wp::BVHPackedNodeHalf));

        std::vector<int> _parents(bvh.max_nodes);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _parents.data(), bvh.node_parents, bvh.max_nodes * sizeof(int));

        std::vector<int> _num_children(bvh.max_nodes);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _num_children.data(), num_children, bvh.max_nodes * sizeof(int));

        FILE* f = fopen("D:/Work/dev/optix/optix_cluster_bench.clean/_build/warp_log", "w");
        for (uint32_t i = 0; i < (bvh.max_nodes); ++i)
        {
            wp::BVHPackedNodeHalf const& lo = _node_lowers[i], up = _node_uppers[i];
            fprintf(f, "%d (%f %f %f) (%f %f %f) l=%d r=%d b=%d p=%d cc=%d\n", i, lo.x, lo.y, lo.z, up.x, up.y, up.z, lo.i, up.i, lo.b, _parents[i], _num_children[i]);
        }
        fclose(f);
    }
#endif         
    profiler[Profiler::Timer::pack_leaves].start((CUstream)WP_CURRENT_CONTEXT);
    wp_launch_device(WP_CURRENT_CONTEXT, mark_packed_leaf_nodes, bvh.max_nodes, (bvh.max_nodes, range_lefts, range_rights, bvh.node_parents, bvh.node_lowers, bvh.node_uppers));
    profiler[Profiler::Timer::pack_leaves].stop();

    profiler[Profiler::Timer::global].stop();

    profiler.print_timers( stdout );

    //debug(bvh);

#if 0
    {
        wp_cuda_context_synchronize(WP_CURRENT_CONTEXT);
        std::vector<wp::BVHPackedNodeHalf> _node_lowers(bvh.max_nodes), _node_uppers(bvh.max_nodes);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _node_lowers.data(), bvh.node_lowers, bvh.max_nodes * sizeof(wp::BVHPackedNodeHalf));
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _node_uppers.data(), bvh.node_uppers, bvh.max_nodes * sizeof(wp::BVHPackedNodeHalf));

        std::vector<int> _parents(bvh.max_nodes);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, _parents.data(), bvh.node_parents, bvh.max_nodes * sizeof(int));

        int max_depth = 0, root = -1;
        for( uint32_t i = 0; i < (uint32_t)bvh.max_nodes; ++i )
        {
            int depth = 1;
            int parent = _parents[i];
            
            if( parent == -1 )
                root = i;

            while (parent != -1)
            {
                parent = _parents[parent];
                depth++;
            }
            max_depth = std::max(max_depth, depth);
        }
        printf("root=%d max_depth=%d nnodes=%d\n", root, max_depth, bvh.max_nodes);

        FILE* f = fopen("D:/Work/dev/optix/optix_cluster_bench.clean/_build/warp_log", "w");
        for (uint32_t i = 0; i < (bvh.max_nodes); ++i)
        {
            wp::BVHPackedNodeHalf const& lo = _node_lowers[i], up = _node_uppers[i];
            fprintf(f, "%d (%f %f %f) (%f %f %f) l=%d r=%d b=%d p=%d\n", i, lo.x, lo.y, lo.z, up.x, up.y, up.z, lo.i, up.i, lo.b, _parents[i]);
        }
        fclose(f);
    }
#endif

    // free temporary memory
    wp_free_device(WP_CURRENT_CONTEXT, indices);
    wp_free_device(WP_CURRENT_CONTEXT, keys);
    wp_free_device(WP_CURRENT_CONTEXT, deltas);

    wp_free_device(WP_CURRENT_CONTEXT, range_lefts);
    wp_free_device(WP_CURRENT_CONTEXT, range_rights);
    wp_free_device(WP_CURRENT_CONTEXT, num_children);    
}

// buffer_size is the number of T, not the number of bytes
template<typename T>
T* make_device_buffer_of(void* context, T* host_buffer, size_t buffer_size)
{
    T* device_buffer = (T*)wp_alloc_device(context, sizeof(T) * buffer_size);;
    wp_memcpy_h2d(context, device_buffer, host_buffer, sizeof(T) * buffer_size);

    return device_buffer;
}

void copy_host_tree_to_device(void* context, BVH& bvh_host, BVH& bvh_device_on_host)
{
#ifdef REORDER_HOST_TREE


    // reorder bvh_host such that its nodes are in the front
    // this is essential for the device refit 
    BVHPackedNodeHalf* node_lowers_reordered = new BVHPackedNodeHalf[bvh_host.max_nodes];
    BVHPackedNodeHalf* node_uppers_reordered = new BVHPackedNodeHalf[bvh_host.max_nodes];

    int* node_parents_reordered = new int[bvh_host.max_nodes];

    std::vector<int> old_to_new(bvh_host.max_nodes, -1);

    // We will place nodes in this order:
    //   Pass 1: leaf nodes (except if it's the root index)
    //   Pass 2: non-leaf, non-root
    //   Pass 3: root node
    int next_pos = 0;

    const int root_index = *bvh_host.root;
    // Pass 1: place leaf nodes at the front 
    for (int i = 0; i < bvh_host.num_nodes; ++i)
    {
        if (bvh_host.node_lowers[i].b)
        {
            node_lowers_reordered[next_pos] = bvh_host.node_lowers[i];
            node_uppers_reordered[next_pos] = bvh_host.node_uppers[i];
            old_to_new[i] = next_pos;
            next_pos++;
        }
    }

    // Pass 2: place non-leaf, non-root nodes
    for (int i = 0; i < bvh_host.num_nodes; ++i)
    {
        if (i == root_index)
        {
            if (bvh_host.node_lowers[i].b)
                // if root node is leaf node, there must be only be one node
            {
                *bvh_host.root = 0;
            }
            else
            {
                *bvh_host.root = next_pos;
            }
        }
        if (!bvh_host.node_lowers[i].b)
        {
            node_lowers_reordered[next_pos] = bvh_host.node_lowers[i];
            node_uppers_reordered[next_pos] = bvh_host.node_uppers[i];
            old_to_new[i] = next_pos;
            next_pos++;
        }
    }

    // We can do that by enumerating all old->new pairs:
    for (int old_index = 0; old_index < bvh_host.num_nodes; ++old_index) {
        int new_index = old_to_new[old_index];  // new index

        int old_parent = bvh_host.node_parents[old_index];
        if (old_parent != -1)
        {
            node_parents_reordered[new_index] = old_to_new[old_parent];
        }
        else
        {
            node_parents_reordered[new_index] = -1;
        }

        // only need to modify the child index of non-leaf nodes
        if (!bvh_host.node_lowers[old_index].b)
        {
            node_lowers_reordered[new_index].i = old_to_new[bvh_host.node_lowers[old_index].i];
            node_uppers_reordered[new_index].i = old_to_new[bvh_host.node_uppers[old_index].i];
        }
    }

    delete[] bvh_host.node_lowers;
    delete[] bvh_host.node_uppers;
    delete[] bvh_host.node_parents;

    bvh_host.node_lowers = node_lowers_reordered;
    bvh_host.node_uppers = node_uppers_reordered;
    bvh_host.node_parents = node_parents_reordered;
#endif // REORDER_HOST_TREE

    bvh_device_on_host.num_nodes = bvh_host.num_nodes;
    bvh_device_on_host.num_leaf_nodes = bvh_host.num_leaf_nodes;
    bvh_device_on_host.max_nodes = bvh_host.max_nodes;
    bvh_device_on_host.num_items = bvh_host.num_items;
    bvh_device_on_host.max_depth = bvh_host.max_depth;

    bvh_device_on_host.root = (int*)wp_alloc_device(context, sizeof(int));
    wp_memcpy_h2d(context, bvh_device_on_host.root, bvh_host.root, sizeof(int));
    bvh_device_on_host.context = context;

    bvh_device_on_host.node_lowers = make_device_buffer_of(context, bvh_host.node_lowers, bvh_host.max_nodes);
    bvh_device_on_host.node_uppers = make_device_buffer_of(context, bvh_host.node_uppers, bvh_host.max_nodes);
    bvh_device_on_host.node_parents = make_device_buffer_of(context, bvh_host.node_parents, bvh_host.max_nodes);
    bvh_device_on_host.primitive_indices = make_device_buffer_of(context, bvh_host.primitive_indices, bvh_host.num_items);
}

// create in-place given existing descriptor
void bvh_create_device(void* context, vec3* lowers, vec3* uppers, int num_items, int constructor_type, BVH& bvh_device_on_host)
{
    ContextGuard guard(context);
    if (constructor_type == BVH_CONSTRUCTOR_SAH || constructor_type == BVH_CONSTRUCTOR_MEDIAN)
        // CPU based constructors
    {
        // copy bounds back to CPU
        std::vector<vec3> lowers_host(num_items);
        std::vector<vec3> uppers_host(num_items);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, lowers_host.data(), lowers, sizeof(vec3) * num_items);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, uppers_host.data(), uppers, sizeof(vec3) * num_items);

        // run CPU based constructor
        wp::BVH bvh_host;
        wp::bvh_create_host(lowers_host.data(), uppers_host.data(), num_items, constructor_type, bvh_host);

        // copy host tree to device
        wp::copy_host_tree_to_device(WP_CURRENT_CONTEXT, bvh_host, bvh_device_on_host);
        // replace host bounds with device bounds
        bvh_device_on_host.item_lowers = lowers;
        bvh_device_on_host.item_uppers = uppers;
        // node_counts is not allocated for host tree
        bvh_device_on_host.node_counts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes);
        wp::bvh_destroy_host(bvh_host);
    }
    else if (constructor_type == BVH_CONSTRUCTOR_LBVH)
    {
        bvh_device_on_host.num_items = num_items;
        bvh_device_on_host.max_nodes = 2 * num_items - 1;
        bvh_device_on_host.num_leaf_nodes = num_items;
        bvh_device_on_host.node_lowers = (BVHPackedNodeHalf*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes);
        wp_memset_device(WP_CURRENT_CONTEXT, bvh_device_on_host.node_lowers, 0, sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes);
        bvh_device_on_host.node_uppers = (BVHPackedNodeHalf*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes);
        wp_memset_device(WP_CURRENT_CONTEXT, bvh_device_on_host.node_uppers, 0, sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes);
        bvh_device_on_host.node_parents = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes);
        bvh_device_on_host.node_counts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes);
        bvh_device_on_host.root = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int));
        bvh_device_on_host.primitive_indices = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * num_items);
        bvh_device_on_host.item_lowers = lowers;
        bvh_device_on_host.item_uppers = uppers;

        bvh_device_on_host.context = context ? context : wp_cuda_context_get_current();

        LinearBVHBuilderGPU builder;
        builder.build(bvh_device_on_host, lowers, uppers, num_items, NULL);
    }
    else
    {
        printf("Unrecognized Constructor type: %d! For GPU constructor it should be SAH (0), Median (1), or LBVH (2)!\n", constructor_type);
    }
}

void bvh_destroy_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);

    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_lowers); bvh.node_lowers = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_uppers); bvh.node_uppers = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_parents); bvh.node_parents = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_counts); bvh.node_counts = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.primitive_indices); bvh.primitive_indices = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.root); bvh.root = NULL;
}


} // namespace wp


void wp_bvh_refit_device(uint64_t id)
{
    wp::BVH bvh;
    if (bvh_get_descriptor(id, bvh))
    {
        ContextGuard guard(bvh.context);

        wp::bvh_refit_device(bvh);
    }
}

/*
* Since we don't even know the number of true leaf nodes, never mention where they are, we will launch
* the num_items threads, which are identical to the number of leaf nodes in the original tree. The 
* refitting threads will start from the nodes corresponding to the original leaf nodes, which might be 
* muted. However, the muted leaf nodes will still have the pointer to their parents, thus the up-tracing
* can still work. We will only compute the bounding box of a leaf node if its parent is not a leaf node.
*/
uint64_t wp_bvh_create_device(void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type)
{
    ContextGuard guard(context);
    wp::BVH bvh_device_on_host;
    wp::BVH* bvh_device_ptr = nullptr;
    
    wp::bvh_create_device(WP_CURRENT_CONTEXT, lowers, uppers, num_items, constructor_type, bvh_device_on_host);

    // create device-side BVH descriptor
    bvh_device_ptr = (wp::BVH*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::BVH));
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, bvh_device_ptr, &bvh_device_on_host, sizeof(wp::BVH));

    uint64_t bvh_id = (uint64_t)bvh_device_ptr;
    wp::bvh_add_descriptor(bvh_id, bvh_device_on_host);
    return bvh_id;
}

uint64_t wp_bvh_debug_device(void* context, char const* filepath)
{
    ContextGuard guard(context);

    int num_items = 0;
    std::vector<wp::vec3> lowers_host, uppers_host;
    wp::vec3 inv;
    {
        if(FILE* f = fopen(filepath, "rb"))
        {
            fread(&num_items, sizeof(int), 1, f);
            fread(&wp::_total_bounds, sizeof(wp::bounds3), 1, f);
            fread(&wp::_total_inv_edges, sizeof(wp::vec3), 1, f);

            lowers_host.resize(num_items);
            fread(lowers_host.data(), sizeof(wp::vec3), num_items, f);        

            uppers_host.resize(num_items);
            fread(uppers_host.data(), sizeof(wp::vec3), num_items, f);

            fclose(f);
            printf("read : '%s' (bounds=%d)\n", filepath, num_items);
        }
        else 
        {
            printf("cannot read : '%s'\n", filepath);
            return 0;
        }
    }

    wp::vec3* lowers = (wp::vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, num_items * sizeof(wp::vec3));
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, lowers, lowers_host.data(), sizeof(wp::vec3) * num_items);

    wp::vec3* uppers = (wp::vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, num_items * sizeof(wp::vec3));
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, uppers, uppers_host.data(), sizeof(wp::vec3) * num_items);

    return wp_bvh_create_device(context, lowers, uppers, num_items, BVH_CONSTRUCTOR_LBVH);
}

void wp_bvh_destroy_device(uint64_t id)
{
    wp::BVH bvh;
    if (wp::bvh_get_descriptor(id, bvh))
    {
        wp::bvh_destroy_device(bvh);
        wp::bvh_rem_descriptor(id);

        // free descriptor
        wp_free_device(WP_CURRENT_CONTEXT, (void*)id);
    }
}
