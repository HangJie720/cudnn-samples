#define IN_SIZE (1*10*1*1)
#define OUT_SIZE (1*10*1*1)
#define TOL (0.000005)

#define CUDNN_DTYPE CUDNN_DATA_FLOAT
typedef float stype;
typedef float dtype;

dtype input[IN_SIZE] = {-1.13631976f,0.99733211f,0.23526242f,0.76501533f,-1.16602302f,-0.07635724f,0.32103113f,1.10676830f,-0.69401034f,0.92491012f};

dtype output[OUT_SIZE] = {0.021245886f,0.17943539f,0.08374240f,0.14223753f,0.020624094f,0.06132121f,0.091241896f,0.20018688f,0.033064913f,0.16689973f};

