;;; 
;;; Copyright (c) 2015, Jan Tatham, All rights reserved.
;;;
;;; Redistribution and use in source and binary forms, with or without
;;; modification, are permitted provided that the following conditions
;;; are met:
;;;
;;; * Redistributions of source code must retain the above copyright
;;; notice, this list of conditions and the following disclaimer.
;;;
;;; * Redistributions in binary form must reproduce the above
;;; copyright notice, this list of conditions and the following
;;; disclaimer in the documentation and/or other materials
;;; provided with the distribution.
;;;
;;; THIS SOFTWARE IS PROVIDED BY THE AUTHOR 'AS IS' AND ANY EXPRESSED
;;; OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
;;; WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
;;; ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
;;; DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
;;; DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
;;; GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
;;; INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
;;; WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
;;; NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;;; SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
;;;

(in-package #:noise)

(declaim (optimize (speed 3) (debug 0)))

(defvar +perm+
  #(151 160 137 91 90 15 131 13 201 95 96 53 194 233 7 225 140 36 103 30 69 142
    8 99 37 240 21 10 23 190 6 148 247 120 234 75 0 26 197 62 94 252 219 203
    117 35 11 32 57 177 33 88 237 149 56 87 174 20 125 136 171 168 68 175 74
    165 71 134 139 48 27 166 77 146 158 231 83 111 229 122 60 211 133 230 220
    105 92 41 55 46 245 40 244 102 143 54 65 25 63 161 1 216 80 73 209 76 132
    187 208 89 18 169 200 196 135 130 116 188 159 86 164 100 109 198 173 186 3
    64 52 217 226 250 124 123 5 202 38 147 118 126 255 82 85 212 207 206 59 227
    47 16 58 17 182 189 28 42 223 183 170 213 119 248 152 2 44 154 163 70 221
    153 101 155 167 43 172 9 129 22 39 253 19 98 108 110 79 113 224 232 178 185
    112 104 218 246 97 228 251 34 242 193 238 210 144 12 191 179 162 241 81 51
    145 235 249 14 239 107 49 192 214 31 181 199 106 157 184 84 204 176 115 121
    50 45 127 4 150 254 138 236 205 93 222 114 67 29 24 72 243 141 128 195 78
    66 215 61 156 180
    151 160 137 91 90 15 131 13 201 95 96 53 194 233 7 225 140 36 103 30 69 142
    8 99 37 240 21 10 23 190 6 148 247 120 234 75 0 26 197 62 94 252 219 203
    117 35 11 32 57 177 33 88 237 149 56 87 174 20 125 136 171 168 68 175 74
    165 71 134 139 48 27 166 77 146 158 231 83 111 229 122 60 211 133 230 220
    105 92 41 55 46 245 40 244 102 143 54 65 25 63 161 1 216 80 73 209 76 132
    187 208 89 18 169 200 196 135 130 116 188 159 86 164 100 109 198 173 186 3
    64 52 217 226 250 124 123 5 202 38 147 118 126 255 82 85 212 207 206 59 227
    47 16 58 17 182 189 28 42 223 183 170 213 119 248 152 2 44 154 163 70 221
    153 101 155 167 43 172 9 129 22 39 253 19 98 108 110 79 113 224 232 178 185
    112 104 218 246 97 228 251 34 242 193 238 210 144 12 191 179 162 241 81 51
    145 235 249 14 239 107 49 192 214 31 181 199 106 157 184 84 204 176 115 121
    50 45 127 4 150 254 138 236 205 93 222 114 67 29 24 72 243 141 128 195 78
    66 215 61 156 180))

(defvar +f2+ 0.366025403d0) ; f2 = 0.5*(sqrt(3.0)-1.0)
(defvar +g2+ 0.211324865d0) ; g2 = (3.0-Math.sqrt(3.0))/6.0
(defvar +f3+ 0.333333334d0) ; f3 = 1.0/3.0
(defvar +g3+ 0.166666667d0) ; g3 = 1.0/6.0


;;;; fast-floor
(defun fast-floor (x)
  (if (> x 0)
      x
      (- x 1)))


;;;; grad-2d fixnum double-float -> double-float
(defun grad-1d (hash x)
  (declare (fixnum hash)
	   (type double-float x))
  (let* ((h (logand hash 15))
	 (grad (+ 1d0 (logand h 7))))
    (declare (fixnum h)
	     (type double-float grad))
    (when (> (logand h 8) 0)
      (setf grad (- grad)))
    
    (* grad x)))


;;;; grad-2d fixnum double-float double-float -> double-float
(defun grad-2d (hash x y)
  (declare (fixnum hash)
	   (type double-float x y))
  (let* ((h (logand hash 7))
	 (u (if (< h 4)
		x
		y))
	 (v (if (< h 4)
		y
		x))
	 (du (if (> (logand h 1) 0)
		 (- u)
		 u))
	 (dv (if (> (logand h 2) 0)
		 (* -2d0 v)
		 (* 2d0 v))))
    (declare (fixnum h)
	     (type double-float u v du dv))
    (+ du dv)))


;;;; grad-3d 
(defun grad-3d (hash x y z)
  (declare (fixnum hash)
	   (type double-float x y z))
  (let* ((h (logand hash 15))
	 (u (if (< h 8)
		x
		y))
	 (v (if (< h 4)
		y
		(if (or (eql h 12) (eql h 14))
		    x
		    z)))
	 (du (if (> (logand h 1) 0)
		 (- u)
		 u))
	 (dv (if (> (logand h 2) 0)
		 (- v)
		 v)))
    (declare (fixnum h)
	     (type double-float u v du dv))
    (+ du dv)))



;;;; noise-1d : double-float -> double-float
;;;; Description: 1D Simplex Noise
(defun noise-1d (x)
  (declare (type double-float x))
  (let ((n0 0.0d0)
	(n1 0.0d0)
	(i0 (floor (fast-floor (floor x)))))
    (declare (type double-float n0 n1)
	     (fixnum i0))
    (let ((i1 (floor (+ i0 1)))
	  (x0 (- x i0)))
      (declare (fixnum i1)
	       (type double-float x0))
      (let ((x1 (- x0 1d0)))
	(declare (type double-float x1))
	(let ((t0 (- 1d0 (* x0 x0)))
	      (t1 (- 1d0 (* x1 x1))))
	  (declare (type double-float t0 t1))
	  
	  (setf t0 (* t0 t0))
	  (setf n0 (* t0 t0 (grad-1d (aref +perm+ (logand i0 #xff)) x0)))

	  (setf t1 (* t1 t1))
	  (setf n1 (* t1 t1 (grad-1d (aref +perm+ (logand i1 #xff)) x1)))

	  (* 0.25d0 (+ n0 n1)))))))


;;;; noise-2d : double-float double-float -> double-float
;;;; Description: 2D Simplex Noise
(defun noise-2d (x y)
  (declare (type double-float x y))
  (let ((n0 0.0d0)
	(n1 0.0d0)
	(n2 0.0d0)
	(i1 0)
	(j1 0)
	(s (* (+ x y) +f2+)))
    (declare (fixnum i1 j1)
	     (type double-float n0 n1 n2 s))

    (let ((i (floor (fast-floor (+ x s))))
	  (j (floor (fast-floor (+ y s)))))
      (declare (fixnum i j))

      (let ((tx (* (+ i j) +g2+)))
	(declare (type double-float tx))

	(let ((x0 (- x (- i tx)))
	      (y0 (- y (- j tx))))
	  (declare (type double-float x0 y0))
	  (if (> x0 y0)
	      (setf i1 1)
	      (setf i1 0))
	  (if (> x0 y0)
	      (setf j1 0)
	      (setf j1 1))

	  (let ((x1 (+ (- x0 i1) +g2+))
		(y1 (+ (- y0 j1) +g2+))
		(x2 (- x0 (+ 1d0 (* 2d0 +g2+))))
		(y2 (- y0 (+ 1d0 (* 2d0 +g2+))))
		(ii (logand i 255))
		(jj (logand j 255)))
	    (declare (type double-float x1 y1 x2 y2)
		     (fixnum ii jj))

	    (let ((gi0 (aref +perm+ (+ ii (aref +perm+ jj))))
		  (gi1 (aref +perm+ (+ ii i1 (aref +perm+ (+ jj j1)))))
		  (gi2 (aref +perm+ (+ ii 1 (aref +perm+ (+ jj 1)))))
		  (t0 (- 0.5d0 (* x0 x0) (* y0 y0)))
		  (t1 (- 0.5d0 (* x1 x1) (* y1 y1)))
		  (t2 (- 0.5d0 (* x2 x2) (* y2 y2))))
	      (declare (fixnum gi0 gi1 gi2)
		       (type double-float t0 t1 t2))

	      (if (< t0 0)
		  (setf n0 0d0)
		  (progn (setf t0 (* t0 t0))
			 (setf n0 (* t0 t0 (grad-2d gi0 x0 y0)))))
	      
	      (if (< t1 0)
		  (setf n1 0d0)
		  (progn (setf t1 (* t1 t1))
			 (setf n1 (* t1 t1 (grad-2d gi1 x1 y1)))))
	      
	      (if (< t2 0)
		  (setf n2 0d0)
		  (progn (setf t2 (* t2 t2))
			 (setf n2 (* t2 t2 (grad-2d gi2 x2 y2)))))
	      
	      (* 70d0 (+ n0 n1 n2)))))))))



;;;; noise-3d : double-float double-float double-float -> double-float
;;;; Description: 3D Simplex Noise
(defun noise-3d (x y z)
  (declare (type double-float x y))
  (let ((n0 0.0d0)
	(n1 0.0d0)
	(n2 0.0d0)
	(n3 0.0d0)
	(i1 0)
	(j1 0)
	(k1 0)
	(i2 0)
	(j2 0)
	(k2 0)
	(s (* (+ x y z) +f3+)))
    (declare (type double-float n0 n1 n2 n3 s)
	     (fixnum i1 j2 k1 i2 j2 k2))
    (let ((i (floor (fast-floor (+ x s))))
	  (j (floor (fast-floor (+ y s))))
	  (k (floor (fast-floor (+ z s)))))
      (declare (fixnum i j k))

      (let ((tx (* (+ i j k) +g3+)))
	(declare (type double-float tx))

	(let ((x0 (- x (- i tx)))
	      (y0 (- y (- j tx)))
	      (z0 (- z (- k tx))))
	  (declare (type double-float x0 y0 z0))

	  (if (>= x0 y0)
	      (if (>= y0 z0)
		  (progn (setf i1 1) (setf j1 0) (setf k1 0)
			 (setf i2 1) (setf j2 1) (setf k2 0)) ; X Y Z Order
		  (if (>= x0 z0)
		      (progn (setf i1 1) (setf j1 0) (setf k1 0)
			     (setf i2 1) (setf j2 0) (setf k2 1)) ; X Z Y Order
		      (progn (setf i1 0) (setf j1 0) (setf k1 1)
			     (setf i2 1) (setf j2 0) (setf k2 1)))) ; Z X Y order
	      (if (< y0 z0)
		  (progn (setf i1 0) (setf j1 0) (setf k1 1)
			 (setf i2 0) (setf j2 1) (setf k2 1)) ; Z Y X Order
		  (if (< x0 z0)
		      (progn (setf i1 0) (setf j1 1) (setf k1 0)
			     (setf i2 0) (setf j2 1) (setf k2 1)) ; Y Z Z Order
		      (progn (setf i1 0) (setf j1 1) (setf k1 0)
			     (setf i2 1) (setf j2 1) (setf k2 0))))) ; Y X Z order
	  (let ((x1 (- x0 (+ i1 +g3+)))
		(y1 (- y0 (+ j1 +g3+)))
		(z1 (- z0 (+ k1 +g3+)))
		(x2 (- x0 (+ i2 (* 2d0 +g3+))))
		(y2 (- y0 (+ j2 (* 2d0 +g3+))))
		(z2 (- z0 (+ k2 (* 2d0 +g3+))))
		(x3 (- x0 (+ 1d0 (* 3d0 +g3+))))
		(y3 (- y0 (+ 1d0 (* 3d0 +g3+))))
		(z3 (- z0 (+ 1d0 (* 3d0 +g3+))))
		(ii (logand i #xff))
		(jj (logand j #xff))
		(kk (logand k #xff)))
	    (declare (fixnum ii jj kk)
		     (type double-float x1 y1 z1 x2 y2 z2 x3 y3 z3))

	    (let ((gi0 (aref +perm+ (+ ii (aref +perm+ (+ jj (aref +perm+ kk))))))
		  (gi1 (aref +perm+ (+ ii i1 (aref +perm+ (+ jj j1 (aref +perm+ (+ kk k1)))))))
		  (gi2 (aref +perm+ (+ ii i2 (aref +perm+ (+ jj j2 (aref +perm+ (+ kk k2)))))))
		  (gi3 (aref +perm+ (+ ii 1 (aref +perm+ (+ jj 1 (aref +perm+ (+ kk 1)))))))
		  (t0 (- 0.6d0 (* x0 x0) (* y0 y0) (* z0 z0)))
		  (t1 (- 0.6d0 (* x1 x1) (* y1 y1) (* z1 z1)))
		  (t2 (- 0.6d0 (* x2 x2) (* y2 y2) (* z2 z2)))
		  (t3 (- 0.6d0 (* x3 x3) (* y3 y3) (* z3 z3))))
	      (declare (fixnum gi0 gi1 gi2 gi3)
		       (type double-float t0 t1 t2 t3))

	      (if (< t0 0d0)
		  (setf n0 0d0)
		  (progn (setf t0 (* t0 t0))
			 (setf n0 (* t0 t0 (grad-3d gi0 x0 y0 z0)))))

	      (if (< t1 0d0)
		  (setf n1 0d0)
		  (progn (setf t1 (* t1 t1))
			 (setf n1 (* t1 t1 (grad-3d gi1 x1 y1 z1)))))
	      
	      (if (< t2 0d0)
		  (setf n2 0d0)
		  (progn (setf t2 (* t2 t2))
			 (setf n2 (* t2 t2 (grad-3d gi2 x2 y2 z2)))))

	      (if (< t3 0d0)
		  (setf n3 0d0)
		  (progn (setf t3 (* t3 t3))
			 (setf n3 (* t3 t3 (grad-3d gi3 x3 y3 z3)))))

	      (* 32d0 (+ n0 n1 n2 n3)))))))))
