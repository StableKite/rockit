#pragma once

#include <list>
#include "RTTaskNodeDef.h"
#include "rt_header.h"

class RTTaskNode;
typedef RTTaskNode *(*RTNodeCreate)();
typedef struct _RTNodeStub {
    const INT32 mUid;
    const char *mName;
    const char *mVersion;
    RTNodeCreate mCreateObj;
    RTPadCaps mCapsSrc;
    RTPadCaps mCapsSink;
} RTNodeStub;

class RTTaskNodeFactory {
 public:
    RTTaskNodeFactory();
    virtual ~RTTaskNodeFactory();
    static RTTaskNodeFactory *instance();
    RT_RET registerStub(RTNodeStub *stub);
    RTTaskNode *createNode(const char *name);
    RTTaskNode *createNode(RTStubUid stubKey);
    RTTaskNode *createNode(RTPadCaps *caps);
    RTTaskNode *createNode(RTNodeStub *stub);
    RTStubUid getUidByName(const char *name);
    RTNodeStub *getStubByUid(RTStubUid stubKey);
 private:
    std::list<RTNodeStub *> mStubs;
};

class RTTaskNodeToken {
 public:
    explicit RTTaskNodeToken(const RTNodeStub &stub);
    virtual ~RTTaskNodeToken();
 private:
    RTNodeStub *mNodeStub{nullptr};
};
