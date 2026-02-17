variable "DOCKERHUB_REPO" {
  default = "dtometzki"
}

variable "DOCKERHUB_IMG" {
  default = "run_vllm"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

variable "HUGGINGFACE_ACCESS_TOKEN" {
  default = ""
}

group "default" {
  targets = ["worker-vllm"]
}

target "worker-vllm" {
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}"]
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
}
