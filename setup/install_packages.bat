copy /y requirements.txt d:\python37\ai\Scripts
cd d:\python37\ai\Scripts
activate & pip install -r requirements.txt --trusted-host mirrors.aliyun.com -i https://mirrors.aliyun.com/pypi/simple/ & deactivate
