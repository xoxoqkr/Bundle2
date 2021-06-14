# -*- coding: utf-8 -*-

import simpy
import numpy as np
import random

## 현재 env에 있는 모든 process를 쭉 접근할 수 있는 방법이 없음.
## 따라서, 매번 프로세스를 따로 리스트의 형태로 저장해주는 것이 필요함.
current_ps = []


class Sample_process:
    def __init__(self, env, i, n, run_t):
        self.env = env
        self.rider = simpy.Resource(env, capacity = 1)
        self.name = i
        self.target = n
        self.end = run_t
        self.pool = list(range(run_t))
        self.run = env.process(self.Running())
    
    def Running(self):
        print('런 타입',type(self.run))
        while True:
            print('잔존 수', self.pool)
            if len(self.pool) > 0:
                time = self.pool[0]
                with self.rider.request() as req:
                    yield req
                    print('리퀘스트 타입',type(req))
                    try:
                        yield env.process(self.sinlge_process(time, self.target))
                        self.pool.remove(time)
                    except simpy.Interrupt:
                        print('now {} interrupted! {} ~ {} removed'.format( env.now , time, time + self.target - 1))
                        input('확인2')
                        for i in range(time, min(time + self.target, self.end)):
                            self.pool.remove(i)
                        #req.interrupt()
                        #return env.timeout(2)
                        #return env.process(self.Running())
                        #yield env.timeout(2)
                        #return env.timeout(1)
                        return None
            else:
                print('종료 됨', env.now)
                break
            input('확인1')


    def sinlge_process(self, time,n):
        try:
            print('T {} start {}'.format(int(env.now), time))
            yield env.timeout(time)
            print('T {} done {}'.format(int(env.now), time))
        except simpy.Interrupt:
            print('T {} pass {}'.format(int(env.now), time))
            for i in range(time, time + n):
                self.pool.remove(i)
            yield env.timeout(time + n)
            print('T {} move to {}'.format(int(env.now), time + n))


def stop_any_process(env, process_list):
    ## 2초마다 한번씩 현재 process 중 아무거나 종료시키는 generator
    ## 남아있는 clock이 없을때의 조건도 만들어줌.
    while True:
        try:
            yield env.timeout(3)
            print("취소 시도 ::{}".format(env.now))
            r = np.random.randint(0, len(process_list))
            print('선택된',r)
            print("취소 시도1 ::{}".format(env.now))
            #process_list[r].interrupt()
            process_list[r].run.interrupt()
            print("취소 시도2 ::{}".format(env.now))
            env.process(process_list[r].Running())
            #process_list[r].rider.users[0].interrupt()
            #process_list[r].Running.interrupt()
            #process_list[r].run.sinlge_process.interrupt()
            print("취소 시도3 ::{}".format(env.now))
            #current_ps.remove(current_ps[r])
        except:
            yield env.timeout(3)
            print("#" * 20)
            print("all process was interrupted at {}".format(env.now))


def stop_any_process2(env, process_list):
    ## 2초마다 한번씩 현재 process 중 아무거나 종료시키는 generator
    ## 남아있는 clock이 없을때의 조건도 만들어줌.
    while True:
        yield env.timeout(3)
        rv = 0.2
        if rv > random.random():
            print('취소 현재', env.now)
            print("취소 시도 ::{}".format(env.now))
            r = np.random.randint(0, len(process_list))
            print('선택된',r,'확인', process_list[r])
            input('멈춤')
            print("취소 시도1 ::{}".format(env.now))
            if len(process_list[r].pool) > 0:
                process_list[r].run.interrupt()
                print("취소 시도2 ::{}".format(env.now))
                process_list[r].run = env.process(process_list[r].Running())
                #env.process(process_list[r].Running())
                print("취소 시도3 ::{}".format(env.now))
            else:
                print('종료2')
                break
            #current_ps.remove(current_ps[r])
        else:
            pass

## environment setting
env = simpy.Environment()

## 6 개의 중간에 멈출 수 있는 clock을 만들어서 집어넣음
process_list = []
p = Sample_process(env, 1, 5, 26)
process_list.append(p)

## 2초마다 process를 멈추는 generator도 넘겨줌
#env.process(stop_any_process(env,process_list))
env.process(stop_any_process2(env,process_list))
env.run(until=200)

input('테스트 종료 됨')

"""
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
"""