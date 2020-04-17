clang++ -std=c++11 -g --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/cuda \
-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -IFreeImage/include \
-LFreeImage/lib/linux/x86_64 -LFreeImage/lib/linux -lcublas -lcudnn -lfreeimage -lstdc++ -lm \
-o test_bi test.cu