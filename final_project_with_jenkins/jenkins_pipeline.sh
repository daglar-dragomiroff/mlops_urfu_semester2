echo "*** DATASETS CREATION *** "
/home/radmil/MLOps/venv_mlops/bin/python /home/radmil/MLOps/venv_mlops/final_project_with_jenkins/create_datasets.py
echo "*** MODEL CREATION ***"
/home/radmil/MLOps/venv_mlops/bin/python /home/radmil/MLOps/venv_mlops/final_project_with_jenkins/create_model.py
echo "*** MODEL TESTING WITH PYTEST ***"
/home/radmil/MLOps/venv_mlops/bin/python /home/radmil/MLOps/venv_mlops/final_project_with_jenkins/test_model_with_pytest.py