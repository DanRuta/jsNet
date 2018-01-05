#include "gmock/gmock.h"

class MockLayer : public Layer {
public:

    MockLayer (int netI, int s);

    virtual ~MockLayer();

    MOCK_METHOD1(assignNext, void(Layer* l));

    MOCK_METHOD1(assignPrev, void(Layer* l));

    MOCK_METHOD1(init, void(int layerIndex));

    MOCK_METHOD0(forward, void(void));

    MOCK_METHOD1(backward, void(bool lastLayer));

    MOCK_METHOD0(applyDeltaWeights, void(void));

    MOCK_METHOD0(resetDeltaWeights, void(void));
};