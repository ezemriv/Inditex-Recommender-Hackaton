# Set aliases
git config alias.pushmain '!git push origin main && git push github main'
git config alias.pushdevelop '!git push origin develop && git push github develop'

# Push to bitbucket and github
git pushmain
git pushdevelop
