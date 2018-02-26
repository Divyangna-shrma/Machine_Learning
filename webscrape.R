library(rvest)
ht <- read_html('https://www.google.co.in/search?q=shimla')
links <- ht %>% html_nodes(xpath='//h3/a') %>% html_attr('href')
sites<-gsub('/url\\?q=','',sapply(strsplit(links[as.vector(grep('url',links))],split='&'),'[',1))
sites
for (i in sites){
  content<-read_html(i)
  title <- content %>% html_nodes('p') %>% html_text()
  print(title)
}