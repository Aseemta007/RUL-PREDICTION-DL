# How to Upload Your Project to GitHub - Step by Step Guide

## Prerequisites
- ✅ Git is installed (you have version 2.43.0)
- ✅ A GitHub account (if you don't have one, create it at github.com)
- ✅ Your project files are ready

## Step 1: Initialize Git Repository (Already Done!)
Your repository has been initialized. The `.gitignore` file is also created to exclude unnecessary files like:
- Virtual environment folders (`haha/`)
- Python cache files (`__pycache__/`)
- Other temporary files

## Step 2: Stage Your Files
Add all your project files to git staging area:
```powershell
git add .
```
This tells git which files you want to include in your commit.

**Note:** The `.gitignore` file will automatically exclude files/folders that shouldn't be uploaded (like your virtual environment).

## Step 3: Create Your First Commit
Commit your files with a descriptive message:
```powershell
git commit -m "Initial commit: CALCE RUL Deep Learning Project"
```
This saves a snapshot of your project. The `-m` flag adds a commit message.

## Step 4: Create a Repository on GitHub
1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: e.g., "CALCE-RUL-Project" or "mini-with-calce-data"
   - **Description**: (optional) Brief description of your project
   - **Visibility**: Choose **Public** or **Private**
   - **DO NOT** check "Initialize with README" (since you already have files)
5. Click **"Create repository"**

## Step 5: Connect Your Local Repository to GitHub
After creating the repository, GitHub will show you commands. You'll need to:

1. **Add the remote repository:**
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```
   Replace `YOUR_USERNAME` with your GitHub username and `YOUR_REPO_NAME` with the repository name you created.

2. **Rename your main branch (if needed):**
   ```powershell
   git branch -M main
   ```
   This ensures your branch is named "main" (GitHub's default).

## Step 6: Push Your Code to GitHub
Upload your code:
```powershell
git push -u origin main
```
The `-u` flag sets up tracking so future pushes can be done with just `git push`.

**Note:** You'll be prompted for your GitHub username and password. 
- For password, you may need to use a **Personal Access Token** instead of your regular password (GitHub requires this for security)
- To create a token: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token

## Step 7: Verify Your Upload
Go to your GitHub repository page and refresh. You should see all your files there!

---

## Future Updates

When you make changes to your project and want to upload them:

1. **Check what changed:**
   ```powershell
   git status
   ```

2. **Add changed files:**
   ```powershell
   git add .
   ```
   Or add specific files: `git add filename.py`

3. **Commit changes:**
   ```powershell
   git commit -m "Description of what you changed"
   ```

4. **Push to GitHub:**
   ```powershell
   git push
   ```

---

## Common Commands Reference

| Command | What it does |
|---------|-------------|
| `git status` | Shows which files have changed |
| `git add .` | Stages all changes |
| `git add filename` | Stages a specific file |
| `git commit -m "message"` | Saves changes with a message |
| `git push` | Uploads to GitHub |
| `git pull` | Downloads latest from GitHub |
| `git log` | Shows commit history |

---

## Troubleshooting

**Problem:** "GitHub authentication failed"
- **Solution:** Use a Personal Access Token instead of password. Create one at: GitHub Settings → Developer settings → Personal access tokens

**Problem:** "Repository not found"
- **Solution:** Check that the repository name and your username in the remote URL are correct

**Problem:** "Branch 'main' does not exist"
- **Solution:** Run `git branch -M main` to rename your branch

---

## Quick Checklist

- [ ] Git repository initialized ✅ (Already done!)
- [ ] .gitignore file created ✅ (Already done!)
- [ ] Stage files with `git add .`
- [ ] Create commit with `git commit -m "message"`
- [ ] Create repository on GitHub website
- [ ] Add remote with `git remote add origin [URL]`
- [ ] Push with `git push -u origin main`

Good luck! 🚀

