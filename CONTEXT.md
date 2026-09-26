# CANDID-PTX task-set experiments

These runs train and score pneumothorax on one chest radiograph collection, CANDID-PTX, under different combinations of its three tasks.

## Language

**CANDID-PTX**:
The chest radiograph collection these experiments use. A study on it can carry a pneumothorax label, boxes, and a mask.
_Avoid_: dataset, when you mean one task or one task set

**Task**:
One annotation type on CANDID-PTX: classification, localization, or segmentation.
_Avoid_: dataset, head

**Task set**:
The subset of those three tasks that one run trains. Experiments (a)–(g) are the seven task sets.
_Avoid_: dataset combination, cyclictask

**Cycle**:
One pass through the task set. A one-task task set has one epoch per cycle; a three-task task set has three.
_Avoid_: epoch, when you mean a full pass

**Smoke run**:
One cycle of the full task set, kept in its own output directory, used to check that each task trains and that all three scores are written.
_Avoid_: debug job, test epoch

**Pretraining run**:
A task-set run that starts from the Ark encoder with untrained task heads.
_Avoid_: finetune

**Patient index**:
The leading token of a reports filename, the text before the first underscore. Images that share it belong to one patient.
_Avoid_: StudyInstanceUID, SOPInstanceUID

**Pneumothorax mask**:
The run-length encoding of pneumothorax on one study. An empty mask is `-1`.
_Avoid_: rib-fracture mask, bounding box

**Classification label**:
`1` when that study's pneumothorax mask is present, and `0` when the mask is empty.
_Avoid_: box presence, report text

**Partition**:
The train, validation, and test assignment already stored with the localization boxes, shared by classification, localization, and segmentation.
_Avoid_: split file, redraw

**Test partition**:
The studies in the localization test assignment. Their scores are focused and unfocused performance.
_Avoid_: validation

**Student**:
The model whose weights the optimizer updates.
_Avoid_: online model

**Teacher**:
The exponential-moving-average copy of the student. It is scored every epoch and the optimizer does not update it.
_Avoid_: EMA, when you mean this role

**Training trajectory**:
The held-out score of each task after every epoch, recorded for the student and for the teacher.
_Avoid_: training loss

**Focused performance**:
The test-partition score of a task, measured on the checkpoint written immediately after an epoch that trained that same task.
_Avoid_: dataset, when an Ark+ note says Dataset D

**Unfocused performance**:
The test-partition score of a task, measured on the checkpoint written immediately after an epoch that trained a different task.
_Avoid_: average score, cycle score
