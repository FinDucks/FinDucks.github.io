#[포켓몬데이터입력쿼리]
DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
number int,
name varchar(20),
type varchar(10),
height float,
weight float
);
INSERT INTO mypokemon (number, name, type, height, weight)
VALUES (10, 'caterpie', 'bug', 0.3, 2.9),
(25, 'pikachu', 'electric', 0.4, 6),
(26, 'raichu', 'electric', 0.8, 30),
(125, 'electabuzz', 'electric', 1.1, 30),
(133, 'eevee', 'normal', 0.3, 6.5),
(137, 'porygon', 'normal', 0.8, 36.5),
(152, 'chikoirita', 'grass', 0.9, 6.4),
(153, 'bayleef', 'grass', 1.2, 15.8),
(172, 'pichu', 'electric', 0.3, 2),
(470, 'leafeon', 'grass', 1, 25.5);
/*MISSION (1)
포켓몬테이블에서이름의길이가5보다큰포켓몬들을타입(type)을기준으로그룹화하고,
몸무게(weight)의평균이20 이상인그룹의타입과, 몸무게의평균을가져와주세요. 이때, 결과는
몸무게의평균을내림차순으로정렬해주세요.*/

select type, avg(weight) from mypokemon 
where (length(name) >5) 
group by type 
having avg(weight) > 20
order by avg(weight) desc;

/*
포켓몬테이블에서번호(number)가200보다작은포켓몬들을타입(type)을기준으로그룹화한후에,
몸무게(weight)의최댓값이10보다크거나같고최솟값은2보다크거나같은그룹의
타입, 키(height)의최솟값, 최댓값을가져와주세요. 이때, 결과는키의최솟값의내림차순으로정렬해
주시고, 만약키의최솟값이같다면키의최댓값의내림차순으로정렬해주세요.
*/
select type, min(height), max(height) from mypokemon
where number < 200
group by type 
having max(weight) >=10 and min(weight) >= 2
order by min(height) desc, max(height) desc;

# [포켓몬데이터입력쿼리]
DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
number int,
name varchar(20),
type varchar(10),
height float,
weight float
);
INSERT INTO mypokemon (number, name, type, height, weight)
VALUES (10, 'caterpie', 'bug', 0.3, 2.9),
(25, 'pikachu', 'electric', 0.4, 6),
(26, 'raichu', 'electric', 0.8, 30),
(125, 'electabuzz', 'electric', 1.1, 30),
(133, 'eevee', 'normal', 0.3, 6.5),
(137, 'porygon', 'normal', 0.8, 36.5),
(152, 'chikoirita', 'grass', 0.9, 6.4),
(153, 'bayleef', 'grass', 1.2, 15.8),
(172, 'pichu', 'electric', 0.3, 2),
(470, 'leafeon', 'grass', 1, 25.5);

/*
포켓몬타입별키의평균을가져와주세요.
*/
select type, avg(height) from mypokemon group by type;

/*
포켓몬의타입별몸무게의평균을가져와주세요.
*/
select type, avg(weight) from mypokemon group by type;

/*
포켓몬의타입별키의평균과몸무게의평균을함께가져와주세요.
*/
select type, avg(height), avg(weight) from mypokemon
group by type;

/*
키의평균이0.5 이상인포켓몬의타입을가져와주세요.
*/
select type from mypokemon
group by type
having avg(height) >= 0.5;


/*
몸무게의평균이20이상인포켓몬의타입을가져와주세요.
*/
select type from mypokemon
group by type
having avg(weight) >= 20;

/*MISSION (6)
포켓몬의type 별번호(number)의합을가져와주세요.
*/
select type, sum(number) from mypokemon
group by type;

/*MISSION (7)
키가0.5 이상인포켓몬이포켓몬의type 별로몇개씩있는지가져와주세요.
*/
select type, count(1) from mypokemon
where height > 0.5
group by type;


/*MISSION (8)
포켓몬타입별키의최솟값을가져와주세요.
*/
select type, min(height) from mypokemon
group by type;
/*MISSION (9)
포켓몬타입별몸무게의최댓값을가져와주세요.
*/
select type, max(weight) from mypokemon
group by type;
/*MISSION (10)
키의최솟값이0.5보다크고몸무게의최댓값이30보다작은포켓몬타입을가져와주세요
*/
select type from mypokemon
group by type
having max(weight) < 30 and min(height) > 0.5;
/*
*/
