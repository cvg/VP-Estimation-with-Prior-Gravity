set(FOLDER_NAME "estimators")

UNCALIBRATED_VP_ADD_SOURCES(
    bindings.cc
    exhaustive_sampler.h
    vp_ransac.h
    vp_estimator_base.h vp_estimator_base.cpp
    calibrated_3lines.h calibrated_3lines.cpp
    uncalibrated_vp_estimator_base.h uncalibrated_vp_estimator_base.cpp
    uncalibrated_4lines_220.h uncalibrated_4lines_220.cpp
    uncalibrated_4lines_211.h uncalibrated_4lines_211.cpp
    calibrated_vertical_1line.h calibrated_vertical_1line.cpp
    uncalibrated_vertical_vp_estimator_base.h uncalibrated_vertical_vp_estimator_base.cpp
    uncalibrated_vertical_2lines_2.h uncalibrated_vertical_2lines_2.cpp
    uncalibrated_vertical_2lines_11.h uncalibrated_vertical_2lines_11.cpp
    uncalibrated_vertical_2lines_11_v2.h uncalibrated_vertical_2lines_11_v2.cpp
    uncalibrated_vertical_2lines_011.h uncalibrated_vertical_2lines_011.cpp
    functions.h functions.cpp
    gamma_values.cpp

    vp_hybrid_ransac.h
    hybrid_uncalibrated_vp_estimator.h hybrid_uncalibrated_vp_estimator.cpp
    functions_hybrid.h functions_hybrid.cpp
    sprt_preemptive_verification.h
)

