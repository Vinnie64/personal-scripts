#!/bin/sh
dotfiles_script
vim todo
git add todo
git commit -m "Update todo"
git push
suexport
