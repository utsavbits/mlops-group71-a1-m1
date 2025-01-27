# mlops-group71-a1-m1
MLOPS group 71 Assignment -1 M1
## Summary of M1: MLOps Foundation
### Description of the Work Completed
The M1 assignment focused on implementing a CI/CD pipeline for a machine learning project. The tasks included setting up a CI/CD pipeline using GitHub Actions, implementing version control with Git, and deploying a front-end application using Gradio on Hugging Face Space. The pipeline stages included linting, testing, formatting, training, evaluation, and deployment. The project involved creating branches, pull requests, and merging them to trigger the CI/CD pipeline automatically. The final deployment was a drug classification application deployed to Hugging Face Space.
Justification for the Choices Made
1.	**CI/CD Tool (GitHub Actions):** GitHub Actions was chosen for its seamless integration with GitHub repositories and ease of use. It allowed for automated workflows triggered by events such as push and pull requests.
2.	**Linting (Flake8):** Flake8 was used to ensure code quality and adherence to style guidelines by identifying syntax errors early, To generate a code quality report for future improvements and not to block the build.
3.	**Testing (Pytest):** Pytest was selected for its powerful features for writing and running tests. It ensures that the code functions as expected and helps in maintaining code reliability.
4.	**Formatting (Black):** Black was used for code formatting to maintain a consistent code style across the project. It automatically formats the code, reducing the chances of style-related issues.
5.	**Deployment (Hugging Face Space)**: Hugging Face Space was chosen for deploying the front-end application due to its free availability and ease of use for hosting machine learning models and applications.
6.	**Version Control (Git):** Used for version control to manage changes, facilitate collaboration, and maintain a history of modifications. Branching, pull requests, and merging ensured a smooth workflow and integration process.
The choices made were aimed at ensuring a robust, automated, and efficient CI/CD pipeline that enhances code quality, reliability, and ease of deployment.


# Deployed Model
Link: https://huggingface.co/spaces/utsavbits/Drug-Classifier


