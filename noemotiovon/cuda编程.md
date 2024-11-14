### CUDA编程架构

* grid 切分为多个 block，block再切为多个小块给每个thread执行计算。
* 每个block只能去到1个SM，1个SM下可以有多个block。

* 在1个SM上，block中的thread被划分成warp执行，每个warp中一般有32个thread。这种调度方式被叫做**SIMT（Single Instruction Multiple Thread）。**warp是单指令多线程（SIMT）模式，一个warp内的所有thread执行相同的指令。warp是nvidia gpu上最小的调度单元
  * 因此，block中的thread数量一般是32的倍数
  * 如果block中的thread数量不能被32整除，那么会存在一个warp中有>32个thread的情况，多出来的thread会被设置成inactive
* 每个thread占用一个SP（cuda core），即1个warp会占用1个SM上的32个SP。