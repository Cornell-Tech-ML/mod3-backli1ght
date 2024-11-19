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