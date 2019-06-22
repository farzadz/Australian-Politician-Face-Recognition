#!/bin/sh

# install google_images_download
pip3 install google_images_download

# install chrome
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt-get update 
sudo apt-get install google-chrome-stable

# install chromedriver
sudo apt-get install chromium-chromedriver

while IFS='' read -r line || [[ -n "$line" ]]; do (rm -r "$line"); done < "removeList.txt"

# download the images from Google Image Search
# using the keywords in the file "preferredName.txt"
googleimagesdownload -kf "preferredName.txt"

# install twitter-photos
pip3 install twitter-photos

# download photos from twitter accounts specified in "twitterList.txt"
while IFS='' read -r line || [[ -n "$line" ]]; do (if [ ! -d "$line" ]; then (twphotos -u "$line") fi;) done < "twitterList.txt"

# process the filenames of downloaded images
# replace whitespaces with "_"
# remove single quotes and title
cd downloads
for f in *\ *; do mv "$f" "${f// /_}"; done
for f in *\'*; do mv "$f" "${f/\'/}"; done
for f in *_MP*; do mv "$f" "${f/_MP/}"; done
for f in *_Senator*; do mv "$f" "${f/_Senator/}"; done

# remove images whose format is not jpg
find . -maxdepth 2 -type f ! -iname \*.jpg -delete

# count images, rename images with an increasing index one by one
for dir in *; do (cd "$dir" && (n=1; for img in *; do mv "$img" "$(printf ""$dir"_%03i.jpg" "$n")"; ((n++)); done)); done;
n=0; for img in *; do mv "$img" "$(printf "%i.jpg" "$n")"; ((n++)); done

# copy extract.py, extractor.py, and detection folder to the
# parent directory of the directory containing images to extract
# faces
for dir in *_*; do (echo "extract_faces_from_dir(\"$dir\")"); done;
python3 extract.py

# display the number of files in each folder in a sorted order
du -a | cut -d/ -f2 | sort | uniq -c | sort -nr

# perform face extraction
# copy files in mtcnn-face-extraction folder to the parent
# directory of the directorys containing images first
for dir in *_*; do (python3 extract.py "$dir"); done;
# delete empty file
for dir in *; do (find "$dir" -size 0 -delete); done;

# perform clustering
# copy files in face-clustering folder to the parent
# directory of the directorys containing images first
for dir in *_*; do (python3 cluster.py "$dir"); done;