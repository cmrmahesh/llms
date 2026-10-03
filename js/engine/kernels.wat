;; Hand-written WebAssembly SIMD matmul kernel (the "GPU kernel" of this page).
;;
;;   out[t, o] = Σ_i x[t, i] · W[o, i]      W is [outDim × inDim], row-major (nn.Linear)
;;
;; All pointers are byte offsets into the module's linear memory, where the model
;; weights live. Two paths:
;;   * 4 tokens at a time: each weight vector loaded once is multiplied against 4
;;     token rows. This is weight reuse, which is what makes prefill compute-bound.
;;   * 1 token at a time (decode, and leftovers): every weight is used once per load.
;; inDim must be a multiple of 8 (true for every Llama-family hidden size).
;;
;; Build: npx wat2wasm kernels.wat -o kernels.wasm
(module
  (memory (export "memory") 1)

  (func $matmul (export "matmul")
    (param $out i32) (param $x i32) (param $w i32)
    (param $T i32) (param $in i32) (param $outD i32)
    (local $o i32) (local $t i32) (local $i i32)
    (local $wrow i32) (local $x0 i32) (local $x1 i32) (local $x2 i32) (local $x3 i32)
    (local $inB i32) (local $T4 i32) (local $wv v128) (local $wv2 v128)
    (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (local $b0 v128) (local $b1 v128) (local $b2 v128) (local $b3 v128)
    (local $outRow i32)

    (local.set $inB (i32.shl (local.get $in) (i32.const 2)))
    (local.set $T4 (i32.and (local.get $T) (i32.const -4)))
    (local.set $o (i32.const 0))
    (block $done_o
      (loop $loop_o
        (br_if $done_o (i32.ge_u (local.get $o) (local.get $outD)))
        (local.set $wrow (i32.add (local.get $w) (i32.mul (local.get $o) (local.get $inB))))

        ;; ---- blocks of 4 tokens ----
        (local.set $t (i32.const 0))
        (block $done_t4
          (loop $loop_t4
            (br_if $done_t4 (i32.ge_u (local.get $t) (local.get $T4)))
            (local.set $x0 (i32.add (local.get $x) (i32.mul (local.get $t) (local.get $inB))))
            (local.set $x1 (i32.add (local.get $x0) (local.get $inB)))
            (local.set $x2 (i32.add (local.get $x1) (local.get $inB)))
            (local.set $x3 (i32.add (local.get $x2) (local.get $inB)))
            (local.set $a0 (v128.const i32x4 0 0 0 0))
            (local.set $a1 (v128.const i32x4 0 0 0 0))
            (local.set $a2 (v128.const i32x4 0 0 0 0))
            (local.set $a3 (v128.const i32x4 0 0 0 0))
            (local.set $b0 (v128.const i32x4 0 0 0 0))
            (local.set $b1 (v128.const i32x4 0 0 0 0))
            (local.set $b2 (v128.const i32x4 0 0 0 0))
            (local.set $b3 (v128.const i32x4 0 0 0 0))
            (local.set $i (i32.const 0))
            (block $done_i4
              (loop $loop_i4
                (br_if $done_i4 (i32.ge_u (local.get $i) (local.get $inB)))
                (local.set $wv (v128.load (i32.add (local.get $wrow) (local.get $i))))
                (local.set $wv2 (v128.load offset=16 (i32.add (local.get $wrow) (local.get $i))))
                (local.set $a0 (f32x4.add (local.get $a0) (f32x4.mul (local.get $wv) (v128.load (i32.add (local.get $x0) (local.get $i))))))
                (local.set $a1 (f32x4.add (local.get $a1) (f32x4.mul (local.get $wv) (v128.load (i32.add (local.get $x1) (local.get $i))))))
                (local.set $a2 (f32x4.add (local.get $a2) (f32x4.mul (local.get $wv) (v128.load (i32.add (local.get $x2) (local.get $i))))))
                (local.set $a3 (f32x4.add (local.get $a3) (f32x4.mul (local.get $wv) (v128.load (i32.add (local.get $x3) (local.get $i))))))
                (local.set $b0 (f32x4.add (local.get $b0) (f32x4.mul (local.get $wv2) (v128.load offset=16 (i32.add (local.get $x0) (local.get $i))))))
                (local.set $b1 (f32x4.add (local.get $b1) (f32x4.mul (local.get $wv2) (v128.load offset=16 (i32.add (local.get $x1) (local.get $i))))))
                (local.set $b2 (f32x4.add (local.get $b2) (f32x4.mul (local.get $wv2) (v128.load offset=16 (i32.add (local.get $x2) (local.get $i))))))
                (local.set $b3 (f32x4.add (local.get $b3) (f32x4.mul (local.get $wv2) (v128.load offset=16 (i32.add (local.get $x3) (local.get $i))))))
                (local.set $i (i32.add (local.get $i) (i32.const 32)))
                (br $loop_i4)))
            ;; out[(t+k)*outD + o] = hsum(a_k + b_k)
            (local.set $outRow (i32.add (local.get $out)
              (i32.shl (i32.add (i32.mul (local.get $t) (local.get $outD)) (local.get $o)) (i32.const 2))))
            (f32.store (local.get $outRow) (call $hsum (f32x4.add (local.get $a0) (local.get $b0))))
            (local.set $outRow (i32.add (local.get $outRow) (i32.shl (local.get $outD) (i32.const 2))))
            (f32.store (local.get $outRow) (call $hsum (f32x4.add (local.get $a1) (local.get $b1))))
            (local.set $outRow (i32.add (local.get $outRow) (i32.shl (local.get $outD) (i32.const 2))))
            (f32.store (local.get $outRow) (call $hsum (f32x4.add (local.get $a2) (local.get $b2))))
            (local.set $outRow (i32.add (local.get $outRow) (i32.shl (local.get $outD) (i32.const 2))))
            (f32.store (local.get $outRow) (call $hsum (f32x4.add (local.get $a3) (local.get $b3))))
            (local.set $t (i32.add (local.get $t) (i32.const 4)))
            (br $loop_t4)))

        ;; ---- remaining tokens one at a time ----
        (block $done_t1
          (loop $loop_t1
            (br_if $done_t1 (i32.ge_u (local.get $t) (local.get $T)))
            (local.set $x0 (i32.add (local.get $x) (i32.mul (local.get $t) (local.get $inB))))
            (local.set $a0 (v128.const i32x4 0 0 0 0))
            (local.set $a1 (v128.const i32x4 0 0 0 0))
            (local.set $i (i32.const 0))
            (block $done_i1
              (loop $loop_i1
                (br_if $done_i1 (i32.ge_u (local.get $i) (local.get $inB)))
                (local.set $a0 (f32x4.add (local.get $a0)
                  (f32x4.mul (v128.load (i32.add (local.get $wrow) (local.get $i)))
                             (v128.load (i32.add (local.get $x0) (local.get $i))))))
                (local.set $a1 (f32x4.add (local.get $a1)
                  (f32x4.mul (v128.load offset=16 (i32.add (local.get $wrow) (local.get $i)))
                             (v128.load offset=16 (i32.add (local.get $x0) (local.get $i))))))
                (local.set $i (i32.add (local.get $i) (i32.const 32)))
                (br $loop_i1)))
            (f32.store
              (i32.add (local.get $out)
                (i32.shl (i32.add (i32.mul (local.get $t) (local.get $outD)) (local.get $o)) (i32.const 2)))
              (call $hsum (f32x4.add (local.get $a0) (local.get $a1))))
            (local.set $t (i32.add (local.get $t) (i32.const 1)))
            (br $loop_t1)))

        (local.set $o (i32.add (local.get $o) (i32.const 1)))
        (br $loop_o))))

  (func $hsum (param $v v128) (result f32)
    (f32.add
      (f32.add (f32x4.extract_lane 0 (local.get $v)) (f32x4.extract_lane 1 (local.get $v)))
      (f32.add (f32x4.extract_lane 2 (local.get $v)) (f32x4.extract_lane 3 (local.get $v)))))
)
