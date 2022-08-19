---
title: git commands2 - branch
date: 2022-08-18 15:42:48
categories :
- Git
tags:
- branch
- gitignore
- cal
---

### README.md
- 프로젝트와 Repository를 설명하는 책의 표지와 같은 문서
- 나와 동료, 이 repo의 사용자를 위한 문서





### .gitignore
- .gitignore 는 git이 파일을 추적할 때, 어떤 파일이나 폴더 등을 추적하지 않도록 명시하기 위해 작성하며, 해당 문서에 작성된 리스트는 수정사항이 발생해도 git이 무시하게 됨.
- 특정 파일 확장자를 무시하거나 이름에 패턴이 존재하는 경우, 또는 특정 디렉토리 아래의 모든 파일을 무시할 수 있음.
- .gitignore: 특정파일 추적을 하고 싶지 않을 경우


    '''shell  
    *.java  
    *.py[cod]  
    '''

- .gitattributes: 파일단위, 디렉토리 별 다른 설정을 부여하고 싶을 경우

    '''shell  
    'Avoid conflicts in pbxproj files'  
    *.pbxproj binary merge=union  
    'Always diff strings files as text'  
    *.strings text diff  
    '''

- https://www.toptal.com/developers/gitignore

### branch
- 분기점을 생성하여 독립적으로 코드를 변경할 수 있도록 도와주는 모델
- branch 명령어
    - Show available local branch  
        $ git branch
    - Show available remote branch  
        $ git branch -r
    - Show available All branch  
        $ git branch -a
    - Create branch
        $ git branch stem
    - Checkout branch  
        $ git checkout stem
    - Create & Checkout branch  
        $ git checkout -b new-stem
    - make changes inside readme.md  
        $ git commit -a -m 'edit readme.md'
        $ git checkout master
    - git checkout => git switch  
    - merge branch  
        $ git merge stem
    - delete branch
        $ git branch -D stem
    - push with specified remote branch
        $ git push origin stem
    - see the difference between two branches
        $ git diff master stem


### Binet

$F_n=\dfrac{\left(\dfrac{1+\sqrt{5}}{2}\right)^n-\left(\dfrac{1-\sqrt{5}}{2}\right)^n}{\sqrt{5}}$