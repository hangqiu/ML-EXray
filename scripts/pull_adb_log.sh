adb shell 'ls /sdcard/edgeml/*_classification' | tr -d '\r' | sed -e 's/^\///' | xargs -n1 adb pull
