#!/bin/sh
pushd ~/dotfiles
git pull
vim todo
git add todo
git commit -m "Update todo"
git push
suexport
popd
