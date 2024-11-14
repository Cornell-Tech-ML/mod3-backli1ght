# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


Task 3.1 diagnostics output
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/yihanzhou/Desktop/Class Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (163) 
--------------------------------------------------------------------------|loop #ID
    def _map(                                                             | 
        out: Storage,                                                     | 
        out_shape: Shape,                                                 | 
        out_strides: Strides,                                             | 
        in_storage: Storage,                                              | 
        in_shape: Shape,                                                  | 
        in_strides: Strides,                                              | 
    ) -> None:                                                            | 
        for i in prange(len(out)):----------------------------------------| #2
            out_index = np.zeros(MAX_DIMS, np.int32)----------------------| #0
            in_index = np.zeros(MAX_DIMS, np.int32)-----------------------| #1
            to_index(i, out_shape, out_index)                             | 
            broadcast_index(out_index, out_shape, in_shape, in_index)     | 
            data = in_storage[index_to_position(in_index, in_strides)]    | 
            out[index_to_position(out_index, out_strides)] = fn(data)     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--2 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #2) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#2).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (172) is hoisted 
out of the parallel loop labelled #2 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (173) is hoisted 
out of the parallel loop labelled #2 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (205)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/yihanzhou/Desktop/Class Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (205) 
-----------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                      | 
        out: Storage,                                                              | 
        out_shape: Shape,                                                          | 
        out_strides: Strides,                                                      | 
        a_storage: Storage,                                                        | 
        a_shape: Shape,                                                            | 
        a_strides: Strides,                                                        | 
        b_storage: Storage,                                                        | 
        b_shape: Shape,                                                            | 
        b_strides: Strides,                                                        | 
    ) -> None:                                                                     | 
        for i in prange(len(out)):-------------------------------------------------| #6
            out_index = np.zeros(MAX_DIMS, np.int32)-------------------------------| #3
            a_index = np.zeros(MAX_DIMS, np.int32)---------------------------------| #4
            b_index = np.zeros(MAX_DIMS, np.int32)---------------------------------| #5
            to_index(i, out_shape, out_index)                                      | 
            broadcast_index(out_index, out_shape, a_shape, a_index)                | 
            broadcast_index(out_index, out_shape, b_shape, b_index)                | 
            a_data = a_storage[index_to_position(a_index, a_strides)]              | 
            b_data = b_storage[index_to_position(b_index, b_strides)]              | 
            out[index_to_position(out_index, out_strides)] = fn(a_data, b_data)    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #6, #3, #4, #5).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--6 is a parallel loop
   +--3 --> rewritten as a serial loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (parallel)
   +--4 (parallel)
   +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (serial)
   +--4 (serial)
   +--5 (serial)


 
Parallel region 0 (loop #6) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#6).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (217) is hoisted 
out of the parallel loop labelled #6 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (218) is hoisted 
out of the parallel loop labelled #6 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (219) is hoisted 
out of the parallel loop labelled #6 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (251)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/yihanzhou/Desktop/Class Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (251) 
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   | 
        out: Storage,                                              | 
        out_shape: Shape,                                          | 
        out_strides: Strides,                                      | 
        a_storage: Storage,                                        | 
        a_shape: Shape,                                            | 
        a_strides: Strides,                                        | 
        reduce_dim: int,                                           | 
    ) -> None:                                                     | 
        for i in prange(len(out)):---------------------------------| #8
            out_index = np.zeros(MAX_DIMS, np.int32)---------------| #7
            to_index(i, out_shape, out_index)                      | 
            o_index = index_to_position(out_index, out_strides)    | 
            for j in range(a_shape[reduce_dim]):                   | 
                a_index = out_index.copy()                         | 
                a_index[reduce_dim] = j                            | 
                pos_a = index_to_position(a_index, a_strides)      | 
                v = fn(a_storage[pos_a], out[o_index])             | 
                out[o_index] = v                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #8, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (serial)


 
Parallel region 0 (loop #8) had 0 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (261) is hoisted 
out of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
/Users/yihanzhou/Desktop/Class Materials/MLE/workspace/mod3-backli1ght/.venv/lib/python3.12/site-packages/numba/core/typed_passes.py:336: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

Parallel loop listing for  Function _tensor_matrix_multiply, /Users/yihanzhou/Desktop/Class Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (274) 
-------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                             | 
    out: Storage,                                                        | 
    out_shape: Shape,                                                    | 
    out_strides: Strides,                                                | 
    a_storage: Storage,                                                  | 
    a_shape: Shape,                                                      | 
    a_strides: Strides,                                                  | 
    b_storage: Storage,                                                  | 
    b_shape: Shape,                                                      | 
    b_strides: Strides,                                                  | 
) -> None:                                                               | 
    """NUMBA tensor matrix multiply function.                            | 
                                                                         | 
    Should work for any tensor shapes that broadcast as long as          | 
                                                                         | 
    ```                                                                  | 
    assert a_shape[-1] == b_shape[-2]                                    | 
    ```                                                                  | 
                                                                         | 
    Optimizations:                                                       | 
                                                                         | 
    * Outer loop in parallel                                             | 
    * No index buffers or function calls                                 | 
    * Inner loop should have no global writes, 1 multiply.               | 
                                                                         | 
                                                                         | 
    Args:                                                                | 
    ----                                                                 | 
        out (Storage): storage for `out` tensor                          | 
        out_shape (Shape): shape for `out` tensor                        | 
        out_strides (Strides): strides for `out` tensor                  | 
        a_storage (Storage): storage for `a` tensor                      | 
        a_shape (Shape): shape for `a` tensor                            | 
        a_strides (Strides): strides for `a` tensor                      | 
        b_storage (Storage): storage for `b` tensor                      | 
        b_shape (Shape): shape for `b` tensor                            | 
        b_strides (Strides): strides for `b` tensor                      | 
                                                                         | 
    Returns:                                                             | 
    -------                                                              | 
        None : Fills in `out`                                            | 
                                                                         | 
    """                                                                  | 
    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0             | 
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0             | 
                                                                         | 
    assert a_shape[-1] == b_shape[-2]                                    | 
                                                                         | 
    iteration_n = a_shape[-1]                                            | 
                                                                         | 
    for i in prange(len(out)):-------------------------------------------| #9
        out_index = out_shape.copy()                                     | 
        to_index(i, out_shape, out_index)                                | 
        o = index_to_position(out_index, out_strides)                    | 
        temp_sum = 0                                                     | 
        for w in range(iteration_n):                                     | 
            a_index = a_shape.copy()                                     | 
            a_tmp_index = out_index.copy()                               | 
            a_tmp_index[-1] = w                                          | 
            broadcast_index(a_tmp_index, out_shape, a_shape, a_index)    | 
            a_pos = index_to_position(a_index, a_strides)                | 
                                                                         | 
            b_index = b_shape.copy()                                     | 
            b_tmp_index = out_index.copy()                               | 
            b_tmp_index[-2] = w                                          | 
            broadcast_index(b_tmp_index, out_shape, b_shape, b_index)    | 
            b_pos = index_to_position(b_index, b_strides)                | 
            temp_sum += a_storage[a_pos] * b_storage[b_pos]              | 
                                                                         | 
        out[o] = temp_sum                                                | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.