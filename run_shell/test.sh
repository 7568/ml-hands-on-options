systemctl status 60019  | awk '{print $1}'| grep  -oP '[0-9]+(?)$'