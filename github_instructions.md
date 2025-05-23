# Pushing to GitHub

Follow these steps to push your House Price Prediction App to GitHub:

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click on the "+" icon in the upper right corner and select "New repository"
3. Enter "house-price-prediction" as the Repository name
4. Add a description (optional): "A machine learning app for predicting house prices"
5. Choose whether to make the repository Public or Private
6. Do NOT initialize the repository with a README, .gitignore, or license (since we already have these files)
7. Click "Create repository"

## Step 2: Connect Your Local Repository to GitHub

After creating the repository, GitHub will show you commands to connect your existing repository. Use the following commands:

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin git@github.com:YOUR_USERNAME/house-price-prediction.git

# Push your code to GitHub
git push -u origin master
```

If you prefer HTTPS authentication instead of SSH, use:

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/house-price-prediction.git

# Push your code to GitHub
git push -u origin master
```

## Step 3: Verify Your Repository

1. Refresh your GitHub repository page
2. You should see all your files and directories uploaded
3. The README.md will be displayed on the main page

## Next Steps

- Set up GitHub Actions for CI/CD (optional)
- Add collaborators to your repository (Settings > Collaborators)
- Enable Issues and Projects for better project management

## Common Issues

If you encounter authentication issues:
- For HTTPS: You may be prompted for your GitHub username and password or personal access token
- For SSH: Ensure your SSH key is added to your GitHub account and SSH agent

If you need to update your code later:
```bash
# Make changes to your code
git add .
git commit -m "Description of changes"
git push
```

## Troubleshooting

If you see an error like "remote origin already exists", run:
```bash
git remote remove origin
```

Then try adding the remote again.

If you see an error about invalid repository name, check for typos in your username or repository name. 