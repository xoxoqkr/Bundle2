# -*- coding: utf-8 -*-

import simpy
import numpy as np

## 현재 env에 있는 모든 process를 쭉 접근할 수 있는 방법이 없음.
## 따라서, 매번 프로세스를 따로 리스트의 형태로 저장해주는 것이 필요함.
current_ps = []


class Sample_process:
    def __init__(self, env, i, n, run_t):
        self.env = env
        self.rider = simpy.Resource(env, capacity = 1)
        self.name = i
        self.target = n
        self.pool = list(range(run_t))

    def Running(self, run_t):
        while True:
            time = env.now
            with self.rider.request() as req:
                yield req
                yield self.sinlge_process(self, time, self.target)
                self.pool.remove(time)


    def sinlge_process(self, time,n):
        try:
            yield env.timeout(time)
            print('T {} done {}'.format(int(env.now), time))
        except simpy.Interrupt:
            print('T {} pass {}'.format(int(env.now), time))
            yield env.timeout(time + n)
            print('T {} move to {}'.format(int(env.now), time + n))


class clock2:
    def __init__(self,env, i, tick):
        self.env = env
        self.name = i
        self.tick = tick
        env.process(self.Running(self.env))

    def Running(self):
        while True:
            yield self.env.process(self.unit_process(self.tick))
            print('clock {} end at {}'.format(i, env.now))

    def unit_process(self, tick):
        try:
            yield env.timeout(tick)
            print('clock {} ticks at {}'.format(i, env.now))
        except simpy.Interrupt:
            print('## the clock {} was interrupted at {}'.format(i, env.now))
            return None






def clock(env, i, tick):
    ## generator에 interrupt 가 발생했을 때 종료하는 조건을 넣어주어야 함
    while True:
        print('clock {} start at {}'.format(i, env.now))
        try:
            yield env.timeout(tick)
            print('clock {} ticks at {}'.format(i, env.now))
        except simpy.Interrupt:
            print('## the clock {} was interrupted at {}'.format(i, env.now))
            #return None
            yield env.timeout(tick)
            print('clock {} ticks at {}'.format(i, env.now))
        print('clock {} end at {}'.format(i, env.now))

def stop_any_process(env):
    ## 2초마다 한번씩 현재 process 중 아무거나 종료시키는 generator
    ## 남아있는 clock이 없을때의 조건도 만들어줌.
    while True:
        try:
            yield env.timeout(3)
            r = np.random.randint(0, len(current_ps))
            current_ps[r].interrupt()
            #current_ps.remove(current_ps[r])
        except:
            print("#" * 20)
            print("all process was interrupted at {}".format(env.now))
            return None


## environment setting
env = simpy.Environment()

## 6 개의 중간에 멈출 수 있는 clock을 만들어서 집어넣음
for i in range(0, 3):
    p = env.process(clock(env, i, 2))
    ## 새롭게 만들어진 프로세스에 대해서 외부에서 접근 방법이 없으므로, 따로 저장해두어야 함
    current_ps.append(p)

## 2초마다 process를 멈추는 generator도 넘겨줌
env.process(stop_any_process(env))

env.run(until=20)