上传到github流程
通过 git init 改造成git仓库
git add .
git commit -m "ddd"

C盘用户目录下有没有.ssh目录
ssh-keygen -t rsa -C "youremail@example.com"
一路回车

登录Github,找到右上角的图标，打开点进里面的Settings，
再选中里面的SSH and GPG KEYS，点击右上角的New SSH key，
然后Title里面随便填，再把刚才id_rsa.pub里面的内容复制到Title下面的Key内容框里面，
最后点击Add SSH key，这样就完成了SSH Key的加密

在Github上创建一个Git仓库。

git remote add origin git@github.com:ytianjao/test2.git

第一次git push -u origin master
之后git push origin master