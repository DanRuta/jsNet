#include <emscripten.h>
#include <vector>
#include <stdint.h>
#include <memory>

int main(int argc, char const *argv[]) {
    return 0;
}

extern "C" {

    /* NetAssembly.ccallArrays */

    EMSCRIPTEN_KEEPALIVE
    float* getSetWASMArray (float *buf1, int buf1Size, int aNumber, float *buf2, int buf2Size) {

        float values[buf1Size];

        for (int i=0; i<buf1Size; i++) {
            values[i] = buf1[i];
        }

        printf("[WASM] Number is %d\n", aNumber);

        for (int b2=0; b2<buf2Size; b2++) {
            for (int b1=0; b1<buf1Size; b1++) {
                values[b1] = values[b1] * buf2[b2];
            }
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    int32_t* get10Nums (void) {

        #include <cstdlib>
        int32_t *values = (int32_t*) std::malloc(sizeof(*values));

        for (int i=0; i<10; i++) {
            values[i] = i+1;
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    int addNums (float *buf, int bufSize) {

        int x = 0;

        for (int i=0; i<bufSize; i++) {
            x+= buf[i];
        }

        return x;
    }

    // The below are used in testing
    EMSCRIPTEN_KEEPALIVE
    int8_t* testHEAP8 (int8_t *buf, int bufSize) {

        int8_t values[bufSize];

        for (int i=0; i<bufSize; i++) {
            values[i] = buf[i] * 2;
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    uint8_t* testHEAPU8 (uint8_t *buf, int bufSize) {

        uint8_t values[bufSize];

        for (int i=0; i<bufSize; i++) {
            values[i] = buf[i] * 2;
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    int16_t* testHEAP16 (int16_t *buf, int bufSize) {

        int16_t values[bufSize];

        for (int i=0; i<bufSize; i++) {
            values[i] = buf[i] * 2;
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    uint16_t* testHEAPU16 (uint16_t *buf, int bufSize) {

        uint16_t values[bufSize];

        for (int i=0; i<bufSize; i++) {
            values[i] = buf[i] * 2;
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    int32_t* testHEAP32 (int32_t *buf, int bufSize) {

        int32_t values[bufSize];

        for (int i=0; i<bufSize; i++) {
            values[i] = buf[i] * 2;
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    uint32_t* testHEAPU32 (uint32_t *buf, int bufSize) {

        uint32_t values[bufSize];

        for (int i=0; i<bufSize; i++) {
            values[i] = buf[i] * 2;
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    float* testHEAPF32 (float *buf, int bufSize) {

        float values[bufSize];

        for (int i=0; i<bufSize; i++) {
            values[i] = buf[i] * 2;
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

    EMSCRIPTEN_KEEPALIVE
    double* testHEAPF64 (double *buf, int bufSize) {

        double values[bufSize];

        for (int i=0; i<bufSize; i++) {
            values[i] = buf[i] * 2;
        }

        auto arrayPtr = &values[0];
        return arrayPtr;
    }

}