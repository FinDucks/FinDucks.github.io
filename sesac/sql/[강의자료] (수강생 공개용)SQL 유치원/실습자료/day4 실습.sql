USE pokemon;
SELECT *
FROM mypokemon;
# ---
/*
MISSION (1)
포켓몬 테이블에서 이름에 ‘i’가 포함되고, 속도가 50 이하인 포켓몬 데이터를 전부 가져와 주세요.
*/
SELECT *
FROM mypokemon
WHERE name LIKE '%i%' AND speed <= 50;

/*
MISSION (2)
포켓몬 테이블에서 공격력이 50 미만이거나, 방어력이 50 미만인 포켓몬의 이름을
‘weak_pokemon’이라는 별명으로 가져와 주세요.
*/

SELECT name AS weak_pokemon
FROM mypokemon
WHERE attack < 50 OR defense < 50;


/*
MISSION (3)
포켓몬 테이블에서 능력치의 합이 150 이상인 포켓몬의 이름과 능력치의 합을 가져와 주세요.
조건 1. 능력치의 합은 공격력, 방어력, 속도의 합을 의미합니다.
조건 2. 능력치의 합은 ‘total’이라는 별명으로 가져와 주세요.
힌트. 숫자형 데이터 타입 컬럼 간에는 연산이 가능합니다.
*/

SELECT name, attack + defense + speed AS total
FROM mypokemon
WHERE attack + defense + speed >= 150;

