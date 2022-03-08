### Prerequisites
* [docker](https://docs.docker.com/engine/install/ubuntu/)
* [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
### Build the docker image
* Image Classifier: 
```bash
sudo docker build -t hangqiu/k8_classifier:latest .
sudo docker run --gpus all -it hangqiu/k8_classifier:latest
```
### Deploy on Kubernetes
Follow [tutorials on google cloud](https://cloud.google.com/community/tutorials/kubernetes-ml-ops) to push into the container registry. Alternatively, follow [this tutorial](https://cloud.google.com/artifact-registry/docs/docker/quickstart) for artifact registry.
```bash
sudo docker tag hangqiu/k8_classifier:latest us-central1-docker.pkg.dev/${ProjectID}/mlexray/k8_classifier:latest
sudo docker push us-central1-docker.pkg.dev/${ProjectID}/mlexray/k8_classifier:latest
```
