//
//  Shaders.h
//

#ifndef Shaders_h
#define Shaders_h

#ifdef __METAL_VERSION__
#define _SWIFT_NAME(name_) __attribute__((swift_name(#name_)))
#else
@import Foundation;
#define _SWIFT_NAME(name) NS_SWIFT_NAME(name)
#endif


struct MainStepFragmentSortingUniforms {
	unsigned long p, k;
};

struct MainFragmentUniforms {
	unsigned long frame_idx _SWIFT_NAME(frameIndex);
};

#endif // Shaders_h
