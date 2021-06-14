# -*- coding: utf-8 -*-

import simpy  ## 시뮬레이션에 사용하기 위한 라이브러리
from simpy.events import AnyOf, AllOf, Event







def generator1(env):
    print('##generator1 start', env.now)
    yield env.timeout(5)
    print('##generator1 end', env.now)


def generator2(env):
    print("generator2 start", env.now)
    yield env.timeout(13)
    print("generator2 end", env.now)


def AND_generators(env, generators):
    """
    - generator를 여러 개가 모두 종료되면 끝나는 generator를 만듬
        - 아래에 보면, 일반적으로 하는 방식인 env.process(generator)가 아니라
        - simpy.events.Process(env, generator)로 처리했음을 알 수 있음.
        - 두 가지 방식은 동일하나, 지금의 방식의 경우, 해당 프로세스를 이벤트로 고려함.
        - 따라서 이벤트를 AllOf의 방식으로 묶을 수 있음.
    """
    yield AllOf(env,
                [simpy.events.Process(env, g) for g in generators]
                )

    print('Whole process doen, T:', int(env.now))
env = simpy.Environment()
"""
- simpy.events.Process(env, generator1(env)) 는 
- env.process(generator1(env)) 와 의미적으로 같음. 
"""
env.process(AND_generators(env,
                           [generator1(env), generator2(env), generator2(env)]
                           ))
env.run(until=20)