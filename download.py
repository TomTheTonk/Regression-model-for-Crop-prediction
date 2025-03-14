import kagglehub

# Download latest version
path = kagglehub.dataset_download("samuelotiattakorah/agriculture-crop-yield")

print("Path to dataset files:", path)