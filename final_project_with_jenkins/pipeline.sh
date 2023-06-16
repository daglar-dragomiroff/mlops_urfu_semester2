#!/bin/sh
python create_datasets.py
python create_model.py
python test_model_with_pytest.py