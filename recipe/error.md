Warning: Unused dummy argument 'dz' at (1) [-Wunused-dummy-argument]
INFO: /usr/local/bin/gfortran -Wall -g -m64 -Wall -g -undefined dynamic_lookup -bundle -shared build/temp.macosx-10.10-x86_64-cpython-310/build/src.macosx-10.10-x86_64-3.10/hn2016_falwa/interpolate_fieldsmodule.o build/temp.macosx-10.10-x86_64-cpython-310/build/src.macosx-10.10-x86_64-3.10/build/src.macosx-10.10-x86_64-3.10/hn2016_falwa/fortranobject.o build/temp.macosx-10.10-x86_64-cpython-310/hn2016_falwa/f90_modules/interpolate_fields.o -L/usr/local/Cellar/gcc/12.2.0/bin/../lib/gcc/current/gcc/x86_64-apple-darwin20/12 -L/usr/local/Cellar/gcc/12.2.0/bin/../lib/gcc/current/gcc/x86_64-apple-darwin20/12/../../.. -L/usr/local/Cellar/gcc/12.2.0/bin/../lib/gcc/current/gcc/x86_64-apple-darwin20/12/../../.. -lgfortran -o build/lib.macosx-10.10-x86_64-cpython-310/hn2016_falwa/interpolate_fields.cpython-310-darwin.so
gfortran: error: -bundle not allowed with -dynamiclib
error: Command "/usr/local/bin/gfortran -Wall -g -m64 -Wall -g -undefined dynamic_lookup -bundle -shared build/temp.macosx-10.10-x86_64-cpython-310/build/src.macosx-10.10-x86_64-3.10/hn2016_falwa/interpolate_fieldsmodule.o build/temp.macosx-10.10-x86_64-cpython-310/build/src.macosx-10.10-x86_64-3.10/build/src.macosx-10.10-x86_64-3.10/hn2016_falwa/fortranobject.o build/temp.macosx-10.10-x86_64-cpython-310/hn2016_falwa/f90_modules/interpolate_fields.o -L/usr/local/Cellar/gcc/12.2.0/bin/../lib/gcc/current/gcc/x86_64-apple-darwin20/12 -L/usr/local/Cellar/gcc/12.2.0/bin/../lib/gcc/current/gcc/x86_64-apple-darwin20/12/../../.. -L/usr/local/Cellar/gcc/12.2.0/bin/../lib/gcc/current/gcc/x86_64-apple-darwin20/12/../../.. -lgfortran -o build/lib.macosx-10.10-x86_64-cpython-310/hn2016_falwa/interpolate_fields.cpython-310-darwin.so" failed with exit status 1
INFO: 
########### EXT COMPILER OPTIMIZATION ###########
INFO: Platform      : 
  Architecture: x64
  Compiler    : clang

CPU baseline  : 
  Requested   : 'min'
  Enabled     : SSE SSE2 SSE3
  Flags       : -msse -msse2 -msse3
  Extra checks: none

CPU dispatch  : 
  Requested   : 'max -xop -fma4'
  Enabled     : SSSE3 SSE41 POPCNT SSE42 AVX F16C FMA3 AVX2 AVX512F AVX512CD AVX512_KNL AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL
  Generated   : none
INFO: CCompilerOpt.cache_flush[817] : write cache to path -> $SRC_DIR/build/temp.macosx-10.10-x86_64-cpython-310/ccompiler_opt_cache_ext.py
Traceback (most recent call last):
  File "/Users/claresyhuang/opt/anaconda3/envs/intel_cython/bin/conda-build", line 11, in <module>
    sys.exit(main())
  File "/Users/claresyhuang/opt/anaconda3/envs/intel_cython/lib/python3.10/site-packages/conda_build/cli/main_build.py", line 593, in main
    execute(sys.argv[1:])
  File "/Users/claresyhuang/opt/anaconda3/envs/intel_cython/lib/python3.10/site-packages/conda_build/cli/main_build.py", line 573, in execute
    outputs = api.build(
  File "/Users/claresyhuang/opt/anaconda3/envs/intel_cython/lib/python3.10/site-packages/conda_build/api.py", line 253, in build
    return build_tree(
  File "/Users/claresyhuang/opt/anaconda3/envs/intel_cython/lib/python3.10/site-packages/conda_build/build.py", line 3799, in build_tree
    packages_from_this = build(
  File "/Users/claresyhuang/opt/anaconda3/envs/intel_cython/lib/python3.10/site-packages/conda_build/build.py", line 2668, in build
    utils.check_call_env(
  File "/Users/claresyhuang/opt/anaconda3/envs/intel_cython/lib/python3.10/site-packages/conda_build/utils.py", line 450, in check_call_env
    return _func_defaulting_env_to_os_environ("call", *popenargs, **kwargs)
  File "/Users/claresyhuang/opt/anaconda3/envs/intel_cython/lib/python3.10/site-packages/conda_build/utils.py", line 426, in _func_defaulting_env_to_os_environ
    raise subprocess.CalledProcessError(proc.returncode, _args)
subprocess.CalledProcessError: Command '['/bin/bash', '-o', 'errexit', '/Users/claresyhuang/opt/anaconda3/envs/intel_cython/conda-bld/hn2016_falwa_1687236129623/work/conda_build.sh']' returned non-zero exit status 1.
(base) CLARE-IMAC27:hn2016_falwa claresyhuang$ 
