#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

    TEST(Hello, world) {
        EXPECT_EQ(true, true);
    }
}

int main (int argc, char** argv) {
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}