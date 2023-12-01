#! /bin/bash

echo "##### Stopping existing [big-data] container #####\n"

docker stop big-data
stop_status=$?

if [ $stop_status -eq 0 ]; then
    echo "\n##### [big-data] container stopped successfully"
else
    echo "\n##### Failed to stop [big-data] container. Exit status: $stop_status"
    exit 1
fi

echo "\n##### Deleting existing [big-data] container #####"

docker rm big-data
rm_status=$?

if [ $rm_status -eq 0 ]; then
    echo "##### [big-data] container deleted successfully"
else
    echo "##### Failed to delete [big-data] container. Exit status: $rm_status"
    exit 1
fi