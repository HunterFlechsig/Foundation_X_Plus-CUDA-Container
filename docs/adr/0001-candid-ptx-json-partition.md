# CANDID-PTX lists follow the localization JSON

The seven CANDID-PTX task-set runs need one shared train, validation, and test assignment, and the official split bundle is not on SOL. The scratch copy already contains `CANDID_PTX_train_1.json`, `CANDID_PTX_valid_1.json`, and `CANDID_PTX_test_1.json`. Classification and segmentation lists are built from those image ids. A new 80/10/10 patient draw was the alternative, and it would have put different studies in the test partition from the boxes these runs localize.
