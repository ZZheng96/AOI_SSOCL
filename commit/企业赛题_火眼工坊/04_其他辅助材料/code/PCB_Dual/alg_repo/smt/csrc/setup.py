import sys, os, subprocess
from setuptools import setup, Extension
import pybind11

# 源文件路径: 相对于此setup.py所在目录
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_FILE = os.path.join(_HERE, '_fast_core.cpp')

# 检测OpenCV include/libs路径
def find_opencv():
    """尝试通过 pkg-config 或 cmake 查找 OpenCV"""
    # 方式1: pkg-config
    try:
        out = subprocess.check_output(
            ['pkg-config', '--cflags', '--libs', 'opencv4'],
            stderr=subprocess.DEVNULL, text=True).strip()
        flags = out.split()
        inc_dirs = []
        lib_dirs = []
        libs = []
        for f in flags:
            if f.startswith('-I'):
                inc_dirs.append(f[2:])
            elif f.startswith('-L'):
                lib_dirs.append(f[2:])
            elif f.startswith('-l'):
                libs.append(f[2:])
        if inc_dirs:
            return inc_dirs, lib_dirs, libs
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 方式2: 尝试 opencv-python 头文件 (pip包)
    try:
        import cv2
        cv2_dir = os.path.dirname(cv2.__file__)
        inc = os.path.join(cv2_dir, 'include')
        if os.path.isdir(inc):
            lib = os.path.join(cv2_dir, 'lib')
            return [inc], [lib] if os.path.isdir(lib) else [], ['opencv_core', 'opencv_imgproc', 'opencv_imgcodecs']
    except ImportError:
        pass

    # 方式3: 默认系统路径
    return ['/usr/include/opencv4'], ['/usr/lib'], ['opencv_core', 'opencv_imgproc', 'opencv_imgcodecs']

inc_dirs, lib_dirs, libs = find_opencv()

ext = Extension(
    '_fast_core',
    sources=[_SRC_FILE],
    include_dirs=[
        pybind11.get_include(),
        *inc_dirs,
    ],
    library_dirs=lib_dirs,
    libraries=libs,
    extra_compile_args=['-std=c++17', '-O3', '-march=native', '-fopenmp'] if sys.platform != 'win32' else ['/O2', '/openmp'],
    extra_link_args=['-fopenmp'] if sys.platform != 'win32' else [],
)

setup(
    name='_fast_core',
    version='0.1',
    description='C++ accelerated solder detection core',
    ext_modules=[ext],
    zip_safe=False,
)
