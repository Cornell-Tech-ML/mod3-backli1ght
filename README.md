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


* diagnostics output
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/yihanzhou/Desktop/Class Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (164) 
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
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (173) is hoisted 
out of the parallel loop labelled #2 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (174) is hoisted 
out of the parallel loop labelled #2 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (206)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/yihanzhou/Desktop/Class Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (206) 
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
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (218) is hoisted 
out of the parallel loop labelled #6 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (219) is hoisted 
out of the parallel loop labelled #6 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (220) is hoisted 
out of the parallel loop labelled #6 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (252)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/yihanzhou/Desktop/Class Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (252) 
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
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (262) is hoisted 
out of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/yihanzhou/Desktop/Class 
Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (275)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/yihanzhou/Desktop/Class Materials/MLE/workspace/mod3-backli1ght/minitorch/fast_ops.py (275) 
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
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------

# Task 3.5
Split 
CPU:
Epoch  0  loss  8.064512868859866 correct 41
Epoch 0 took 4.24 seconds
Epoch  10  loss  5.021276502116382 correct 39
Epoch 10 took 0.24 seconds
Epoch  20  loss  4.334742140937181 correct 44
Epoch 20 took 0.24 seconds
Epoch  30  loss  4.047239101063314 correct 46
Epoch 30 took 0.24 seconds
Epoch  40  loss  3.272659317978221 correct 47
Epoch 40 took 0.24 seconds
Epoch  50  loss  3.7564624972668454 correct 46
Epoch 50 took 0.24 seconds
Epoch  60  loss  3.337175034906526 correct 49
Epoch 60 took 0.24 seconds
Epoch  70  loss  2.957334588496497 correct 48
Epoch 70 took 0.24 seconds
Epoch  80  loss  2.5613530695857554 correct 49
Epoch 80 took 0.25 seconds
Epoch  90  loss  0.8677818572644602 correct 49
Epoch 90 took 0.24 seconds
Epoch  100  loss  0.6158260781513375 correct 49
Epoch 100 took 0.24 seconds
Epoch  110  loss  2.2119083035866383 correct 45
Epoch 110 took 0.26 seconds
Epoch  120  loss  0.8923196236602633 correct 49
Epoch 120 took 0.26 seconds
Epoch  130  loss  1.0593276810022378 correct 50
Epoch 130 took 0.25 seconds
Epoch  140  loss  1.1047049285130595 correct 50
Epoch 140 took 0.25 seconds
Epoch  150  loss  1.9461427110176497 correct 50
Epoch 150 took 0.24 seconds
Epoch  160  loss  1.5261778549120724 correct 49
Epoch 160 took 0.24 seconds
Epoch  170  loss  0.38663624886291065 correct 47
Epoch 170 took 0.24 seconds
Epoch  180  loss  0.45975355294119574 correct 49
Epoch 180 took 0.24 seconds
Epoch  190  loss  0.23270259128423637 correct 49
Epoch 190 took 0.24 seconds
Epoch  200  loss  1.2307096439574858 correct 50
Epoch 200 took 0.24 seconds
Epoch  210  loss  0.3040067474776271 correct 49
Epoch 210 took 0.24 seconds
Epoch  220  loss  0.14948233558202417 correct 48
Epoch 220 took 0.24 seconds
Epoch  230  loss  2.4232717312623757 correct 49
Epoch 230 took 0.24 seconds
Epoch  240  loss  0.6613201613307602 correct 49
Epoch 240 took 0.24 seconds
Epoch  250  loss  0.18744375774508337 correct 50
Epoch 250 took 0.24 seconds
Epoch  260  loss  1.366835099523603 correct 47
Epoch 260 took 0.24 seconds
Epoch  270  loss  0.9975015267597372 correct 49
Epoch 270 took 0.24 seconds
Epoch  280  loss  1.1487744947393475 correct 47
Epoch 280 took 0.24 seconds
Epoch  290  loss  0.3336042299163335 correct 50
Epoch 290 took 0.24 seconds
Epoch  300  loss  0.3233906471136901 correct 50
Epoch 300 took 0.25 seconds
Epoch  310  loss  0.07738543688467832 correct 49
Epoch 310 took 0.24 seconds
Epoch  320  loss  1.8361507155580572 correct 48
Epoch 320 took 0.24 seconds
Epoch  330  loss  0.12351168429870447 correct 49
Epoch 330 took 0.24 seconds
Epoch  340  loss  0.45501645103948746 correct 49
Epoch 340 took 0.24 seconds
Epoch  350  loss  1.4943212483476804 correct 48
Epoch 350 took 0.24 seconds
Epoch  360  loss  1.6050431956008713 correct 48
Epoch 360 took 0.24 seconds
Epoch  370  loss  1.2012427121784153 correct 49
Epoch 370 took 0.25 seconds
Epoch  380  loss  1.1405150111872833 correct 50
Epoch 380 took 0.24 seconds
Epoch  390  loss  0.16385041487192817 correct 48
Epoch 390 took 0.24 seconds
Epoch  400  loss  0.5453379932489646 correct 50
Epoch 400 took 0.24 seconds
Epoch  410  loss  1.4964761960649473 correct 50
Epoch 410 took 0.24 seconds
Epoch  420  loss  0.32862118657217637 correct 49
Epoch 420 took 0.24 seconds
Epoch  430  loss  0.8803345372699671 correct 50
Epoch 430 took 0.24 seconds
Epoch  440  loss  1.0784797768622056 correct 49
Epoch 440 took 0.24 seconds
Epoch  450  loss  0.21437662403974372 correct 49
Epoch 450 took 0.24 seconds
Epoch  460  loss  0.033652059139494796 correct 50
Epoch 460 took 0.24 seconds
Epoch  470  loss  0.46426145671906427 correct 49
Epoch 470 took 0.26 seconds
Epoch  480  loss  0.548933172926579 correct 50
Epoch 480 took 0.24 seconds
Epoch  490  loss  1.096164435499177 correct 49
Epoch 490 took 0.24 seconds

GPU:
Epoch  0  loss  6.770630174738972 correct 32
Epoch 0 took 6.65169 seconds
Epoch  10  loss  5.067554919637436 correct 40
Epoch 10 took 2.36969 seconds
Epoch  20  loss  4.24556332135686 correct 48
Epoch 20 took 2.34404 seconds
Epoch  30  loss  3.385748771893645 correct 47
Epoch 30 took 2.31407 seconds
Epoch  40  loss  2.278122334050485 correct 44
Epoch 40 took 2.30155 seconds
Epoch  50  loss  3.2864672578574146 correct 50
Epoch 50 took 2.33172 seconds
Epoch  60  loss  1.2543162708057336 correct 49
Epoch 60 took 2.30422 seconds
Epoch  70  loss  1.869115383534887 correct 50
Epoch 70 took 2.30347 seconds
Epoch  80  loss  1.5428898296015146 correct 50
Epoch 80 took 2.31314 seconds
Epoch  90  loss  1.1968715506802003 correct 50
Epoch 90 took 2.28156 seconds
Epoch  100  loss  1.6903231743549831 correct 50
Epoch 100 took 2.46448 seconds
Epoch  110  loss  1.0801929905031595 correct 48
Epoch 110 took 2.65603 seconds
Epoch  120  loss  1.232271635570909 correct 50
Epoch 120 took 2.81237 seconds
Epoch  130  loss  1.1487482226488537 correct 50
Epoch 130 took 3.06611 seconds
Epoch  140  loss  1.1292593443007761 correct 50
Epoch 140 took 3.17564 seconds
Epoch  150  loss  0.8257054857643674 correct 50
Epoch 150 took 2.87303 seconds
Epoch  160  loss  1.0286615520485742 correct 50
Epoch 160 took 2.68043 seconds
Epoch  170  loss  0.28061886476949716 correct 50
Epoch 170 took 2.37070 seconds
Epoch  180  loss  0.5051337497090721 correct 50
Epoch 180 took 2.41360 seconds
Epoch  190  loss  0.7230903944968274 correct 50
Epoch 190 took 2.31865 seconds
Epoch  200  loss  0.6354402991286946 correct 50
Epoch 200 took 2.38051 seconds
Epoch  210  loss  0.2896592774180326 correct 50
Epoch 210 took 2.32529 seconds
Epoch  220  loss  0.9587510181062124 correct 50
Epoch 220 took 3.22039 seconds
Epoch  230  loss  0.16537698321434915 correct 50
Epoch 230 took 2.33590 seconds
Epoch  240  loss  0.4809440969596753 correct 50
Epoch 240 took 2.38570 seconds
Epoch  250  loss  0.2896625150433798 correct 50
Epoch 250 took 2.32709 seconds
Epoch  260  loss  0.3166241362254327 correct 50
Epoch 260 took 2.38589 seconds
Epoch  270  loss  0.29275774502885266 correct 50
Epoch 270 took 2.30176 seconds
Epoch  280  loss  0.7497341644447193 correct 50
Epoch 280 took 2.52404 seconds
Epoch  290  loss  0.15830847028252515 correct 50
Epoch 290 took 2.74528 seconds
Epoch  300  loss  0.2824994956936084 correct 50
Epoch 300 took 3.17064 seconds
Epoch  310  loss  0.10630464280898268 correct 50
Epoch 310 took 3.11646 seconds
Epoch  320  loss  0.11370274546629654 correct 50
Epoch 320 took 2.86520 seconds
Epoch  330  loss  0.27095087378387656 correct 50
Epoch 330 took 2.52024 seconds
Epoch  340  loss  0.10365520648984214 correct 50
Epoch 340 took 2.36140 seconds
Epoch  350  loss  0.2984963638413627 correct 50
Epoch 350 took 2.30946 seconds
Epoch  360  loss  0.16290647275560982 correct 50
Epoch 360 took 2.37831 seconds
Epoch  370  loss  0.3434356391168465 correct 50
Epoch 370 took 2.29501 seconds
Epoch  380  loss  0.15364379766145242 correct 50
Epoch 380 took 2.29683 seconds
Epoch  390  loss  0.10313296545232503 correct 50
Epoch 390 took 2.32465 seconds
Epoch  400  loss  0.05137958509424438 correct 50
Epoch 400 took 2.31468 seconds
Epoch  410  loss  0.045674556074141115 correct 50
Epoch 410 took 2.30404 seconds
Epoch  420  loss  0.05443417314350074 correct 50
Epoch 420 took 2.31871 seconds
Epoch  430  loss  0.09986747720764708 correct 50
Epoch 430 took 2.31320 seconds
Epoch  440  loss  0.2345658333449886 correct 50
Epoch 440 took 2.36524 seconds
Epoch  450  loss  0.24204507362365693 correct 50
Epoch 450 took 2.36632 seconds
Epoch  460  loss  0.26872391596540596 correct 50
Epoch 460 took 2.57568 seconds
Epoch  470  loss  0.04056108222948625 correct 50
Epoch 470 took 2.75122 seconds
Epoch  480  loss  0.2877577096230858 correct 50
Epoch 480 took 3.11213 seconds
Epoch  490  loss  0.17747775941839147 correct 50
Epoch 490 took 3.10760 seconds

Simple:
CPU:
Epoch  0  loss  5.883439236489669 correct 47
Epoch 0 took 3.94216 seconds
Epoch  10  loss  1.5232213825145486 correct 50
Epoch 10 took 0.25193 seconds
Epoch  20  loss  0.528641108703072 correct 50
Epoch 20 took 0.23712 seconds
Epoch  30  loss  0.4621879185171738 correct 49
Epoch 30 took 0.23848 seconds
Epoch  40  loss  1.1393349600603242 correct 50
Epoch 40 took 0.23799 seconds
Epoch  50  loss  1.0756103876436187 correct 50
Epoch 50 took 0.23868 seconds
Epoch  60  loss  0.7475774026635642 correct 50
Epoch 60 took 0.23907 seconds
Epoch  70  loss  1.3672513118434926 correct 50
Epoch 70 took 0.23838 seconds
Epoch  80  loss  2.2040246628148825 correct 48
Epoch 80 took 0.23807 seconds
Epoch  90  loss  0.9298022046639529 correct 50
Epoch 90 took 0.23909 seconds
Epoch  100  loss  1.0940630667131888 correct 50
Epoch 100 took 0.23861 seconds
Epoch  110  loss  0.46516958282775156 correct 50
Epoch 110 took 0.24007 seconds
Epoch  120  loss  0.7185997925815748 correct 50
Epoch 120 took 0.24286 seconds
Epoch  130  loss  0.46218852957887274 correct 50
Epoch 130 took 0.23905 seconds
Epoch  140  loss  0.0015214300662377215 correct 50
Epoch 140 took 0.23761 seconds
Epoch  150  loss  1.0279880643663641 correct 50
Epoch 150 took 0.23871 seconds
Epoch  160  loss  1.3438689282921457 correct 50
Epoch 160 took 0.25814 seconds
Epoch  170  loss  0.3059309317404402 correct 50
Epoch 170 took 0.23734 seconds
Epoch  180  loss  0.506946104015821 correct 50
Epoch 180 took 0.24059 seconds
Epoch  190  loss  0.2766394217394685 correct 50
Epoch 190 took 0.23716 seconds
Epoch  200  loss  0.35162821789878734 correct 50
Epoch 200 took 0.23918 seconds
Epoch  210  loss  0.3164630509428244 correct 50
Epoch 210 took 0.23886 seconds
Epoch  220  loss  0.36214506481540004 correct 50
Epoch 220 took 0.23820 seconds
Epoch  230  loss  0.6295468580063978 correct 50
Epoch 230 took 0.23766 seconds
Epoch  240  loss  0.33798700778523977 correct 50
Epoch 240 took 0.23981 seconds
Epoch  250  loss  0.05689451894663548 correct 50
Epoch 250 took 0.23797 seconds
Epoch  260  loss  0.16228535389954934 correct 50
Epoch 260 took 0.24435 seconds
Epoch  270  loss  0.0861500076558034 correct 50
Epoch 270 took 0.23817 seconds
Epoch  280  loss  0.606383371812542 correct 50
Epoch 280 took 0.23838 seconds
Epoch  290  loss  0.006047419329371741 correct 50
Epoch 290 took 0.23743 seconds
Epoch  300  loss  0.5809249996836494 correct 50
Epoch 300 took 0.23811 seconds
Epoch  310  loss  0.004653296684601908 correct 50
Epoch 310 took 0.24402 seconds
Epoch  320  loss  0.35669232563012965 correct 50
Epoch 320 took 0.23909 seconds
Epoch  330  loss  0.16363960705755887 correct 50
Epoch 330 took 0.24081 seconds
Epoch  340  loss  0.17503316185958248 correct 50
Epoch 340 took 0.24414 seconds
Epoch  350  loss  0.4774366840818999 correct 50
Epoch 350 took 0.23792 seconds
Epoch  360  loss  0.08369428484084515 correct 50
Epoch 360 took 0.23926 seconds
Epoch  370  loss  0.1851586603033991 correct 50
Epoch 370 took 0.23948 seconds
Epoch  380  loss  0.00327543536598443 correct 50
Epoch 380 took 0.24596 seconds
Epoch  390  loss  0.05712468982473641 correct 50
Epoch 390 took 0.23934 seconds
Epoch  400  loss  0.13041588426138506 correct 50
Epoch 400 took 0.23705 seconds
Epoch  410  loss  0.1676504281518485 correct 50
Epoch 410 took 0.23777 seconds
Epoch  420  loss  0.014718933911227281 correct 50
Epoch 420 took 0.24195 seconds
Epoch  430  loss  0.2969870147817993 correct 50
Epoch 430 took 0.23890 seconds
Epoch  440  loss  0.17353393723786192 correct 50
Epoch 440 took 0.23887 seconds
Epoch  450  loss  0.042445272412429284 correct 50
Epoch 450 took 0.23708 seconds
Epoch  460  loss  0.0035637518630845582 correct 50
Epoch 460 took 0.23953 seconds
Epoch  470  loss  0.23750989598129463 correct 50
Epoch 470 took 0.26752 seconds
Epoch  480  loss  0.16009793422112573 correct 50
Epoch 480 took 0.23847 seconds
Epoch  490  loss  0.34358573466526615 correct 50
Epoch 490 took 0.23924 seconds

GPU:
Epoch  0  loss  6.853001191874908 correct 38
Epoch 0 took 5.31136 seconds
Epoch  10  loss  1.9887228060713111 correct 48
Epoch 10 took 2.33083 seconds
Epoch  20  loss  1.7006975672282287 correct 50
Epoch 20 took 2.30841 seconds
Epoch  30  loss  2.4603184047511824 correct 45
Epoch 30 took 2.32296 seconds
Epoch  40  loss  2.079705767794441 correct 50
Epoch 40 took 2.48577 seconds
Epoch  50  loss  1.3353848233022092 correct 48
Epoch 50 took 2.84029 seconds
Epoch  60  loss  1.2566017492662407 correct 50
Epoch 60 took 3.08610 seconds
Epoch  70  loss  1.3218037092265211 correct 50
Epoch 70 took 2.73823 seconds
Epoch  80  loss  0.7137531668190864 correct 50
Epoch 80 took 2.31368 seconds
Epoch  90  loss  0.5772251491918798 correct 50
Epoch 90 took 2.32852 seconds
Epoch  100  loss  0.6504295653131749 correct 50
Epoch 100 took 2.32272 seconds
Epoch  110  loss  1.318791981715209 correct 50
Epoch 110 took 2.34667 seconds
Epoch  120  loss  0.37070909794313367 correct 49
Epoch 120 took 2.31698 seconds
Epoch  130  loss  0.4525840084313577 correct 50
Epoch 130 took 2.30920 seconds
Epoch  140  loss  0.06438493506973227 correct 50
Epoch 140 took 2.40594 seconds
Epoch  150  loss  0.772095749191406 correct 50
Epoch 150 took 2.35942 seconds
Epoch  160  loss  0.8511070443356713 correct 50
Epoch 160 took 2.41464 seconds
Epoch  170  loss  0.30101020657272193 correct 50
Epoch 170 took 2.35814 seconds
Epoch  180  loss  0.5238908154137236 correct 50
Epoch 180 took 2.38333 seconds
Epoch  190  loss  0.862457022725331 correct 50
Epoch 190 took 2.44171 seconds
Epoch  200  loss  0.7391841172372201 correct 50
Epoch 200 took 2.82169 seconds
Epoch  210  loss  0.2662602663926493 correct 50
Epoch 210 took 3.11508 seconds
Epoch  220  loss  0.2786522487983973 correct 50
Epoch 220 took 3.18417 seconds
Epoch  230  loss  0.23971899099669636 correct 50
Epoch 230 took 2.76070 seconds
Epoch  240  loss  0.7081786783394068 correct 50
Epoch 240 took 2.43765 seconds
Epoch  250  loss  1.1122611023639726 correct 50
Epoch 250 took 2.31389 seconds
Epoch  260  loss  0.08545794811969816 correct 50
Epoch 260 took 2.39026 seconds
Epoch  270  loss  0.021208346111855868 correct 50
Epoch 270 took 2.30251 seconds
Epoch  280  loss  0.5265591122659171 correct 50
Epoch 280 took 2.39003 seconds
Epoch  290  loss  0.6136266746287465 correct 50
Epoch 290 took 2.33772 seconds
Epoch  300  loss  0.5068249953356748 correct 50
Epoch 300 took 2.41794 seconds
Epoch  310  loss  1.161306569854326 correct 49
Epoch 310 took 2.33009 seconds
Epoch  320  loss  0.4542935032539804 correct 50
Epoch 320 took 2.37804 seconds
Epoch  330  loss  0.09059448716430847 correct 50
Epoch 330 took 2.33065 seconds
Epoch  340  loss  0.3679657391530259 correct 50
Epoch 340 took 2.45149 seconds
Epoch  350  loss  0.481453806273224 correct 50
Epoch 350 took 2.33809 seconds
Epoch  360  loss  0.07774724753470241 correct 50
Epoch 360 took 2.83831 seconds
Epoch  370  loss  0.6141296913428071 correct 50
Epoch 370 took 3.10320 seconds
Epoch  380  loss  0.0239980776799438 correct 50
Epoch 380 took 3.11377 seconds
Epoch  390  loss  0.0130255900616112 correct 50
Epoch 390 took 2.84899 seconds
Epoch  400  loss  0.06488303787150292 correct 50
Epoch 400 took 2.46929 seconds
Epoch  410  loss  0.764168979467011 correct 50
Epoch 410 took 2.34153 seconds
Epoch  420  loss  0.0006628298798179305 correct 50
Epoch 420 took 2.32220 seconds
Epoch  430  loss  0.036551720823180235 correct 50
Epoch 430 took 2.30766 seconds
Epoch  440  loss  0.0018270562896806303 correct 50
Epoch 440 took 2.38403 seconds
Epoch  450  loss  0.02562933500960495 correct 50
Epoch 450 took 2.31274 seconds
Epoch  470  loss  0.17950733881579733 correct 50
Epoch 470 took 2.33353 seconds
Epoch  480  loss  0.007827239307642932 correct 50
Epoch 480 took 2.29627 seconds
Epoch  490  loss  0.17950733880960493 correct 50
Epoch 490 took 2.29766 seconds

Xor:
CPU: 
Epoch  0  loss  8.066181890500848 correct 30
Epoch 0 took 4.05467 seconds
Epoch  10  loss  5.427726076909687 correct 47
Epoch 10 took 0.23784 seconds
Epoch  20  loss  3.2674689924406306 correct 48
Epoch 20 took 0.23989 seconds
Epoch  30  loss  1.472989863067726 correct 48
Epoch 30 took 0.23741 seconds
Epoch  40  loss  2.2383907619814725 correct 48
Epoch 40 took 0.23642 seconds
Epoch  50  loss  1.390820919264776 correct 48
Epoch 50 took 0.24075 seconds
Epoch  60  loss  3.2086765025930295 correct 47
Epoch 60 took 0.23749 seconds
Epoch  70  loss  3.507498097617551 correct 48
Epoch 70 took 0.23729 seconds
Epoch  80  loss  3.083276402977775 correct 48
Epoch 80 took 0.23711 seconds
Epoch  90  loss  1.1641895452810642 correct 48
Epoch 90 took 0.23599 seconds
Epoch  100  loss  0.36223093489583125 correct 48
Epoch 100 took 0.23645 seconds
Epoch  110  loss  0.7919810995374514 correct 48
Epoch 110 took 0.23695 seconds
Epoch  120  loss  0.8307793160476132 correct 48
Epoch 120 took 0.23766 seconds
Epoch  130  loss  1.7961250204924006 correct 49
Epoch 130 took 0.23710 seconds
Epoch  140  loss  1.1525243691869451 correct 48
Epoch 140 took 0.23647 seconds
Epoch  150  loss  1.5638026653216324 correct 48
Epoch 150 took 0.24023 seconds
Epoch  160  loss  0.8557107347409468 correct 49
Epoch 160 took 0.24359 seconds
Epoch  170  loss  4.053467413378039 correct 46
Epoch 170 took 0.23738 seconds
Epoch  180  loss  0.45281268249969403 correct 49
Epoch 180 took 0.23723 seconds
Epoch  190  loss  1.1960214738050474 correct 49
Epoch 190 took 0.23832 seconds
Epoch  200  loss  1.3102910288840535 correct 49
Epoch 200 took 0.23710 seconds
Epoch  210  loss  0.36076555524607995 correct 48
Epoch 210 took 0.23703 seconds
Epoch  220  loss  1.2077689599024155 correct 48
Epoch 220 took 0.23697 seconds
Epoch  230  loss  0.6229317669930745 correct 49
Epoch 230 took 0.23532 seconds
Epoch  240  loss  0.7737532592139023 correct 49
Epoch 240 took 0.23771 seconds
Epoch  250  loss  1.359451123751835 correct 49
Epoch 250 took 0.23740 seconds
Epoch  260  loss  0.3505166866095586 correct 49
Epoch 260 took 0.23623 seconds
Epoch  270  loss  0.6615262228435987 correct 49
Epoch 270 took 0.23797 seconds
Epoch  280  loss  0.2675102132636098 correct 49
Epoch 280 took 0.23725 seconds
Epoch  290  loss  1.055572624070347 correct 49
Epoch 290 took 0.23663 seconds
Epoch  300  loss  3.073508167333692 correct 47
Epoch 300 took 0.23903 seconds
Epoch  310  loss  0.9819119448681496 correct 49
Epoch 310 took 0.23824 seconds
Epoch  320  loss  0.7511522098095309 correct 49
Epoch 320 took 0.23682 seconds
Epoch  330  loss  0.23385803992268706 correct 49
Epoch 330 took 0.23727 seconds
Epoch  340  loss  2.1701731702026974 correct 49
Epoch 340 took 0.23904 seconds
Epoch  350  loss  0.31506403124834026 correct 49
Epoch 350 took 0.23659 seconds
Epoch  360  loss  0.14475345742140067 correct 49
Epoch 360 took 0.23562 seconds
Epoch  370  loss  2.0260175473401736 correct 49
Epoch 370 took 0.23767 seconds
Epoch  380  loss  1.0930808412276878 correct 49
Epoch 380 took 0.23696 seconds
Epoch  390  loss  0.7908346802709394 correct 49
Epoch 390 took 0.23782 seconds
Epoch  400  loss  0.23167149963341954 correct 49
Epoch 400 took 0.23726 seconds
Epoch  410  loss  0.6491510425634184 correct 49
Epoch 410 took 0.23924 seconds
Epoch  420  loss  0.22654555083062175 correct 49
Epoch 420 took 0.23778 seconds
Epoch  430  loss  0.633503228141093 correct 49
Epoch 430 took 0.23700 seconds
Epoch  440  loss  0.006144671825742856 correct 49
Epoch 440 took 0.24096 seconds
Epoch  450  loss  0.6555975244148732 correct 49
Epoch 450 took 0.23636 seconds
Epoch  460  loss  0.9968085369102545 correct 49
Epoch 460 took 0.23751 seconds
Epoch  470  loss  0.06437199938616893 correct 49
Epoch 470 took 0.23667 seconds
Epoch  480  loss  0.11112443701107569 correct 49
Epoch 480 took 0.23611 seconds
Epoch  490  loss  0.32332276650339625 correct 49
Epoch 490 took 0.23756 seconds

GPU:
Epoch  0  loss  6.129392787556009 correct 26
Epoch 0 took 6.73765 seconds
Epoch  10  loss  3.927729456624434 correct 46
Epoch 10 took 2.40114 seconds
Epoch  20  loss  4.426069455109947 correct 48
Epoch 20 took 2.37668 seconds
Epoch  30  loss  2.9075986330280132 correct 48
Epoch 30 took 2.39149 seconds
Epoch  40  loss  2.3041869207810626 correct 50
Epoch 40 took 2.40083 seconds
Epoch  50  loss  1.810594432933401 correct 49
Epoch 50 took 2.37199 seconds
Epoch  60  loss  2.0461580803376416 correct 50
Epoch 60 took 2.34588 seconds
Epoch  70  loss  2.783700388002001 correct 50
Epoch 70 took 2.71249 seconds
Epoch  80  loss  1.986208430599372 correct 50
Epoch 80 took 3.18244 seconds
Epoch  90  loss  2.216985816451384 correct 50
Epoch 90 took 2.41264 seconds
Epoch  100  loss  1.285440387999263 correct 49
Epoch 100 took 2.38603 seconds
Epoch  110  loss  0.7412461752250237 correct 50
Epoch 110 took 2.33300 seconds
Epoch  120  loss  0.8109555142271647 correct 50
Epoch 120 took 2.35556 seconds
Epoch  130  loss  1.0170844996256194 correct 50
Epoch 130 took 2.37050 seconds
Epoch  140  loss  0.8266028142138244 correct 50
Epoch 140 took 2.42431 seconds
Epoch  150  loss  0.5492140686755688 correct 50
Epoch 150 took 2.32488 seconds
Epoch  160  loss  0.7897219889410387 correct 50
Epoch 160 took 2.43135 seconds
Epoch  170  loss  1.7008171730563362 correct 50
Epoch 170 took 2.50289 seconds
Epoch  180  loss  0.7646558843462581 correct 50
Epoch 180 took 3.05462 seconds
Epoch  190  loss  0.9507688855352239 correct 50
Epoch 190 took 3.12438 seconds
Epoch  200  loss  0.39395480177635817 correct 50
Epoch 200 took 2.97914 seconds
Epoch  210  loss  0.8867136217885023 correct 50
Epoch 210 took 2.35861 seconds
Epoch  220  loss  0.3500976166287607 correct 50
Epoch 220 took 2.40757 seconds
Epoch  230  loss  0.3366680879332819 correct 50
Epoch 230 took 2.43313 seconds
Epoch  240  loss  0.3260095728171302 correct 50
Epoch 240 took 2.48966 seconds
Epoch  250  loss  0.7831864406886488 correct 50
Epoch 250 took 2.35213 seconds
Epoch  260  loss  0.7224085533880459 correct 50
Epoch 260 took 2.36324 seconds
Epoch  270  loss  0.6679705923351918 correct 50
Epoch 270 took 2.78199 seconds
Epoch  280  loss  0.06809375033279524 correct 50
Epoch 280 took 3.10504 seconds
Epoch  290  loss  0.30810289265744145 correct 50
Epoch 290 took 2.74324 seconds
Epoch  300  loss  0.4661851841517308 correct 50
Epoch 300 took 2.41678 seconds
Epoch  310  loss  0.2147085369186416 correct 50
Epoch 310 took 2.33069 seconds
Epoch  320  loss  0.3852441068712895 correct 50
Epoch 320 took 2.40086 seconds
Epoch  330  loss  0.21926934930599082 correct 50
Epoch 330 took 2.30859 seconds
Epoch  340  loss  0.18361805193427555 correct 50
Epoch 340 took 2.38334 seconds
Epoch  350  loss  0.3028273387850157 correct 50
Epoch 350 took 2.31177 seconds
Epoch  360  loss  0.4158921666109767 correct 50
Epoch 360 took 2.37003 seconds
Epoch  370  loss  0.15151697581457368 correct 50
Epoch 370 took 2.28651 seconds
Epoch  380  loss  0.08959963455403502 correct 50
Epoch 380 took 2.50588 seconds
Epoch  390  loss  0.30984951089354096 correct 50
Epoch 390 took 2.86111 seconds
Epoch  400  loss  0.2268627541609708 correct 50
Epoch 400 took 3.01479 seconds
Epoch  410  loss  0.013867848654395543 correct 50
Epoch 410 took 2.73731 seconds
Epoch  420  loss  0.11248713382529045 correct 50
Epoch 420 took 2.30203 seconds
Epoch  430  loss  0.25393575815456015 correct 50
Epoch 430 took 2.30816 seconds
Epoch  440  loss  0.16026051300989955 correct 50
Epoch 440 took 2.35334 seconds
Epoch  450  loss  0.123953278166721 correct 50
Epoch 450 took 2.31835 seconds
Epoch  460  loss  0.18710501608864152 correct 50
Epoch 460 took 2.38284 seconds
Epoch  470  loss  0.18553830097129576 correct 50
Epoch 470 took 2.37508 seconds
Epoch  480  loss  0.1621980295813039 correct 50
Epoch 480 took 2.41610 seconds
Epoch  490  loss  0.23791038404170606 correct 50
Epoch 490 took 2.37047 seconds

Bigger model: hidden_size = 200 on split data
CPU:
Epoch  0  loss  28.29268214692315 correct 30
Epoch 0 took 4.68254 seconds
Epoch  10  loss  4.477688380981764 correct 44
Epoch 10 took 0.72495 seconds
Epoch  20  loss  5.458480012087502 correct 36
Epoch 20 took 0.75148 seconds
Epoch  30  loss  1.8889134036727067 correct 40
Epoch 30 took 0.67645 seconds
Epoch  40  loss  2.2647694021675377 correct 48
Epoch 40 took 0.68300 seconds
Epoch  50  loss  1.4933143836133271 correct 46
Epoch 50 took 0.67955 seconds
Epoch  60  loss  2.392322066066133 correct 46
Epoch 60 took 0.68205 seconds
Epoch  70  loss  1.93778091272232 correct 46
Epoch 70 took 0.68160 seconds
Epoch  80  loss  1.9927815588153075 correct 47
Epoch 80 took 0.68647 seconds
Epoch  90  loss  4.2163791999788724 correct 44
Epoch 90 took 0.68363 seconds
Epoch  100  loss  2.0595302476240693 correct 47
Epoch 100 took 0.68842 seconds
Epoch  110  loss  1.0226991171485804 correct 46
Epoch 110 took 0.68131 seconds
Epoch  120  loss  1.257848985765992 correct 47
Epoch 120 took 0.69271 seconds
Epoch  130  loss  1.7771308312782634 correct 47
Epoch 130 took 0.68132 seconds
Epoch  140  loss  0.28338202837756604 correct 48
Epoch 140 took 0.69548 seconds
Epoch  150  loss  1.187591125665289 correct 47
Epoch 150 took 0.67166 seconds
Epoch  160  loss  1.0987653111028286 correct 47
Epoch 160 took 1.39161 seconds
Epoch  170  loss  1.4608968142919143 correct 47
Epoch 170 took 1.31187 seconds
Epoch  180  loss  0.5496333225105184 correct 49
Epoch 180 took 1.30305 seconds
Epoch  190  loss  0.7535844703902362 correct 47
Epoch 190 took 0.74887 seconds
Epoch  200  loss  1.2305529606964505 correct 48
Epoch 200 took 0.86749 seconds
Epoch  210  loss  1.2675054141394717 correct 47
Epoch 210 took 0.68376 seconds
Epoch  220  loss  1.8013260905203894 correct 48
Epoch 220 took 0.68729 seconds
Epoch  230  loss  1.6574968569156878 correct 48
Epoch 230 took 0.74852 seconds
Epoch  240  loss  0.3697278762471821 correct 48
Epoch 240 took 0.67028 seconds
Epoch  250  loss  1.123120772434797 correct 48
Epoch 250 took 0.67061 seconds
Epoch  260  loss  0.49756200585475885 correct 49
Epoch 260 took 0.66964 seconds
Epoch  270  loss  1.8375055412859909 correct 48
Epoch 270 took 0.67442 seconds
Epoch  280  loss  2.5195230537432245 correct 46
Epoch 280 took 0.67005 seconds
Epoch  290  loss  1.1263377626962663 correct 47
Epoch 290 took 0.67060 seconds
Epoch  300  loss  1.0659945003883395 correct 49
Epoch 300 took 0.66971 seconds
Epoch  310  loss  0.6522055715968459 correct 48
Epoch 310 took 0.66883 seconds
Epoch  320  loss  1.636485325876818 correct 48
Epoch 320 took 0.67356 seconds
Epoch  330  loss  1.4819500621692565 correct 48
Epoch 330 took 0.67279 seconds
Epoch  340  loss  2.923473174861492 correct 47
Epoch 340 took 0.68707 seconds
Epoch  350  loss  0.27851284508134944 correct 50
Epoch 350 took 0.67212 seconds
Epoch  360  loss  0.46444734816772454 correct 49
Epoch 360 took 0.67184 seconds
Epoch  370  loss  3.3795437448728007 correct 46
Epoch 370 took 0.67561 seconds
Epoch  380  loss  1.2965156323351608 correct 49
Epoch 380 took 0.67284 seconds
Epoch  390  loss  0.4893740167409722 correct 49
Epoch 390 took 0.67155 seconds
Epoch  400  loss  0.8360901428132583 correct 48
Epoch 400 took 0.67146 seconds
Epoch  410  loss  0.2835791280512257 correct 49
Epoch 410 took 0.67052 seconds
Epoch  420  loss  0.40519713002567426 correct 48
Epoch 420 took 0.67234 seconds
Epoch  430  loss  0.4155010858876527 correct 50
Epoch 430 took 0.67147 seconds
Epoch  440  loss  0.04636948552338771 correct 47
Epoch 440 took 0.67635 seconds
Epoch  450  loss  1.4754358941652819 correct 49
Epoch 450 took 0.66736 seconds
Epoch  460  loss  0.23102537782945917 correct 50
Epoch 460 took 0.66959 seconds
Epoch  470  loss  0.33445464515670564 correct 49
Epoch 470 took 0.68946 seconds
Epoch  480  loss  0.8630420067473981 correct 49
Epoch 480 took 0.67210 seconds
Epoch  490  loss  0.6192384461504349 correct 49
Epoch 490 took 0.66990 seconds

GPU:
poch  0  loss  13.083898778579751 correct 26
Epoch 0 took 7.74260 seconds
Epoch  10  loss  3.321291415501431 correct 43
Epoch 10 took 4.11476 seconds
Epoch  20  loss  2.633536089430264 correct 47
Epoch 20 took 4.25652 seconds
Epoch  30  loss  1.4488312503377936 correct 46
Epoch 30 took 4.16745 seconds
Epoch  40  loss  3.2843056801193935 correct 47
Epoch 40 took 3.39005 seconds
Epoch  50  loss  1.6653716251582273 correct 48
Epoch 50 took 3.26421 seconds
Epoch  60  loss  2.209276761765677 correct 49
Epoch 60 took 3.32919 seconds
Epoch  70  loss  0.6712976519100428 correct 48
Epoch 70 took 3.33189 seconds
Epoch  80  loss  1.780415271062037 correct 46
Epoch 80 took 3.30065 seconds
Epoch  90  loss  1.154557275633397 correct 48
Epoch 90 took 3.26148 seconds
Epoch  100  loss  2.122454747413464 correct 47
Epoch 100 took 4.01717 seconds
Epoch  110  loss  1.5031319484843682 correct 48
Epoch 110 took 4.18630 seconds
Epoch  120  loss  1.0330550791628688 correct 48
Epoch 120 took 3.41861 seconds
Epoch  130  loss  1.9960711815658636 correct 48
Epoch 130 took 3.14212 seconds
Epoch  140  loss  2.332849474493491 correct 49
Epoch 140 took 3.20089 seconds
Epoch  150  loss  0.7586227668189387 correct 48
Epoch 150 took 3.42199 seconds
Epoch  160  loss  0.16050920542491767 correct 50
Epoch 160 took 4.13813 seconds
Epoch  170  loss  1.7225590453899922 correct 49
Epoch 170 took 3.15632 seconds
Epoch  180  loss  1.2305808579853845 correct 49
Epoch 180 took 3.15670 seconds
Epoch  190  loss  1.706398975104427 correct 48
Epoch 190 took 3.19821 seconds
Epoch  200  loss  1.546647119451537 correct 48
Epoch 200 took 3.30053 seconds
Epoch  210  loss  2.1994576826100833 correct 47
Epoch 210 took 3.67317 seconds
Epoch  220  loss  0.49090184807966475 correct 48
Epoch 220 took 3.99956 seconds
Epoch  230  loss  2.76308373178155 correct 49
Epoch 230 took 3.82936 seconds
Epoch  240  loss  1.295968327336309 correct 50
Epoch 240 took 3.20061 seconds
Epoch  250  loss  1.178733607666734 correct 49
Epoch 250 took 3.24538 seconds
Epoch  260  loss  1.4622515099206213 correct 49
Epoch 260 took 3.23076 seconds
Epoch  270  loss  2.258790309439277 correct 47
Epoch 270 took 3.33504 seconds
Epoch  280  loss  0.14539219078458818 correct 48
Epoch 280 took 3.26515 seconds
Epoch  290  loss  0.5257321853528197 correct 49
Epoch 290 took 3.42678 seconds
Epoch  300  loss  1.7329250092724964 correct 48
Epoch 300 took 3.86824 seconds
Epoch  310  loss  1.9118107280341263 correct 49
Epoch 310 took 4.11278 seconds
Epoch  320  loss  0.15309682291315885 correct 49
Epoch 320 took 4.00142 seconds
Epoch  330  loss  0.20614238782375646 correct 50
Epoch 330 took 3.29772 seconds
Epoch  340  loss  0.929266104147815 correct 48
Epoch 340 took 3.10373 seconds
Epoch  350  loss  0.2335010655706047 correct 48
Epoch 350 took 3.36195 seconds
Epoch  360  loss  1.9698481639751448 correct 48
Epoch 360 took 3.23784 seconds
Epoch  370  loss  0.9155069770459076 correct 49
Epoch 370 took 3.33811 seconds
Epoch  380  loss  1.258578081487463 correct 48
Epoch 380 took 3.40844 seconds
Epoch  390  loss  0.8150129602030648 correct 49
Epoch 390 took 4.10597 seconds
Epoch  400  loss  1.155157507389119 correct 48
Epoch 400 took 4.00695 seconds
Epoch  410  loss  0.06175703847416554 correct 49
Epoch 410 took 3.44271 seconds
Epoch  420  loss  0.09411348466118194 correct 50
Epoch 420 took 3.21248 seconds
Epoch  430  loss  0.8675159352931001 correct 49
Epoch 430 took 3.29689 seconds
Epoch  440  loss  0.8168697925593422 correct 49
Epoch 440 took 3.24767 seconds
Epoch  450  loss  0.4809631239280935 correct 49
Epoch 450 took 3.29170 seconds
Epoch  460  loss  0.5125222561909896 correct 49
Epoch 460 took 3.13289 seconds
Epoch  470  loss  0.6467410363857267 correct 49
Epoch 470 took 4.16038 seconds
Epoch  480  loss  0.8990395032646444 correct 50
Epoch 480 took 3.99963 seconds
Epoch  490  loss  0.7410662482096599 correct 50
Epoch 490 took 3.62423 seconds