#!/bin/bash

#: <<'END'
#END

export OUTPUT_DIR=result_metric
mkdir -p $OUTPUT_DIR

export METRIC= #--no-metric  # TODO remove
export RECORD_CONFIG="--record-mapping-config examples/config/record_mapping/20211105_april.yaml"
export SUBJECT=r_WK37

export MESH=--insole
export MOCAP_CONFIG="--mocap-config examples/config/markers/20210819_april.yaml"
export MANO_CONFIG="--mano-config examples/config/mano/20210610_april.yaml"
export LABEL=close
export DATE=20210819
declare -a SEGMENTS=(12 15 16 17 18)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=-1
        export HAND=mia
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG \
            --demo-file data/${DATE}_april/${DATE}_${SUBJECT}_insole_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $MESH $METRIC
    done
done
export DATE=20210820
declare -a SEGMENTS=(15 15 15 17 18 16 19 20)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=-1
        export HAND=mia
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG \
            --demo-file data/20210819_april/${DATE}_${SUBJECT}_insole_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $MESH $METRIC
    done
done

export MESH=--pillow
export MOCAP_CONFIG="--mocap-config examples/config/markers/20210826_april.yaml"
export MANO_CONFIG="--mano-config examples/config/mano/20210610_april.yaml"
export LABEL=close
export DATE=20210826
declare -a SEGMENTS=(21 21 19 40 19 24)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=-1
        export HAND=mia
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG --mia-thumb-adducted \
            --demo-file data/${DATE}_april/${DATE}_${SUBJECT}_small_pillow_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_pillow_small_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $MESH $METRIC
    done
done
export DATE=20210916
declare -a SEGMENTS=(18 23 19 20)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=-1
        export HAND=mia
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG --mia-thumb-adducted \
            --demo-file data/20210826_april/${DATE}_${SUBJECT}_small_pillow_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_pillow_small_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $MESH $METRIC
    done
done

exit 0

export MESH=--electronic
export MOCAP_CONFIG="--mocap-config examples/config/markers/20211105_april.yaml"
export MANO_CONFIG="--mano-config examples/config/mano/20210610_april.yaml"
export LABEL=grasp
export DATE=20211105
declare -a SEGMENTS=(5 7 6 6 8 8 7 8)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=-1
        export HAND=shadow
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG \
            --demo-file data/${DATE}_april/${DATE}_${SUBJECT}_electronic_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_electronic_grasp_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $MESH $METRIC
    done
done

export MESH=--electronic
export MOCAP_CONFIG="--mocap-config examples/config/markers/20211105_april.yaml"
export MANO_CONFIG="--mano-config examples/config/mano/20210610_april.yaml"
export LABEL=insert
export DATE=20211105
declare -a SEGMENTS=(5 6 6 6 8 8 7 8)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=0
        export HAND=shadow
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG \
            --demo-file data/${DATE}_april/${DATE}_${SUBJECT}_electronic_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_electronic_insert_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $MESH $METRIC
    done
done

export MESH=--passport
export MOCAP_CONFIG="--mocap-config examples/config/markers/20211112_april.yaml"
export MANO_CONFIG="--mano-config examples/config/mano/20210610_april.yaml"
export LABEL=flip
export DATE=20211112
declare -a SEGMENTS=(11 13 14)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=0
        export HAND=mia
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG --mia-thumb-adducted \
            --demo-file data/${DATE}_april/${DATE}_${SUBJECT}_passport_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_passport_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $MESH $METRIC
    done
done

export MOCAP_CONFIG="--mocap-config examples/config/markers/20211126_april_insole.yaml"
export MANO_CONFIG="--mano-config examples/config/mano/20211105_april.yaml"
export LABEL=insert
export DATE=20211126
declare -a SEGMENTS=(6 6)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=0
        export HAND=mia
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG --mia-thumb-adducted \
            --demo-file data/${DATE}_april_insole/${DATE}_${SUBJECT}_insert_insole_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_insole_insert_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $METRIC
    done
done

export MOCAP_CONFIG="--mocap-config examples/config/markers/20211126_april_pillow.yaml"
export MANO_CONFIG="--mano-config examples/config/mano/20211105_april.yaml"
export LABEL=grasp
export DATE=20211126
declare -a SEGMENTS=(33 37 30 30)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=-1
        export HAND=shadow
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG --mia-thumb-adducted \
            --demo-file data/${DATE}_april_pillow/${DATE}_${SUBJECT}_big_pillow_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_pillow_big_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $METRIC
    done
done

export MESH=--passport-closed
export MOCAP_CONFIG="--mocap-config examples/config/markers/20211217_april.yaml"
export MANO_CONFIG="--mano-config examples/config/mano/20211105_april.yaml"
export LABEL=insert
export DATE=20211217
declare -a SEGMENTS=(8 8 9 12)
for i in "${!SEGMENTS[@]}"; do
    for (( j=0; j < SEGMENTS[i]; j++ )); do
        echo "Set ${i}; segment ${j}"
        export SET=${i}
        export SEGMENT=${j}
        export FRAME=0
        export HAND=shadow
        python examples/eval_segment_frame_embodiment.py \
            $HAND $LABEL $SEGMENT $FRAME $MOCAP_CONFIG $MANO_CONFIG $RECORD_CONFIG --mia-thumb-adducted \
            --demo-file data/${DATE}_april/${DATE}_${SUBJECT}_passport_box_set${SET}.json \
            --output-image ${OUTPUT_DIR}/${DATE}_${SET}_passport_insert_${HAND}_${SEGMENT}_${FRAME}.jpg \
            --output-file ${OUTPUT_DIR}/${DATE}_${SET}_insole_${HAND}_${SEGMENT}_${FRAME}.json \
            --show-mano $MESH $METRIC
    done
done
