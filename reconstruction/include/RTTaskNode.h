#pragma once

#include "RTTaskNodeBase.h"

class RTTaskNode : public RTTaskNodeBase {
 public:
    RTTaskNode();
    ~RTTaskNode() override;
    RT_RET invoke(RtMetaData *meta) override;
 protected:
    RT_RET initSupportOptions() override;
    RT_RET invokeInternal(RtMetaData *meta) override;
};
