import wget, zipfile

url_dataset = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
name_zip_dataset = 'kagglecatsanddogs_5340.zip'
directory_to_extract_to = 'dataset_cats_dogs'

wget.download(url_dataset)
with zipfile.ZipFile(name_zip_dataset, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)