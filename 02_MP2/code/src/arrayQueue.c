#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <linux/types.h>
#include <omp.h>

#include "arrayQueue.h"
#include "bitmap.h"

struct ArrayQueue *newArrayQueue(__u32 size){

    struct ArrayQueue* arrayQueue = (struct ArrayQueue*) malloc( sizeof(struct ArrayQueue));
	

    arrayQueue->head = 0;
    arrayQueue->tail = 0;
    arrayQueue->tail_next = 0;
    arrayQueue->size = size;
  


    
    arrayQueue->queue = (__u32*) malloc(size*sizeof(__u32));
	

    arrayQueue->q_bitmap = newBitmap(size);

    arrayQueue->q_bitmap_next = newBitmap(size);

    return arrayQueue;

}


void resetArrayQueue(struct ArrayQueue *q){

	q->head = 0;
    q->tail = 0;
    q->tail_next = 0;
    clearBitmap(q->q_bitmap);

}

void freeArrayQueue(struct ArrayQueue *q){

	freeBitmap(q->q_bitmap_next);
	freeBitmap(q->q_bitmap);

	free(q->queue);
	free(q);

}



void enArrayQueueWithBitmap (struct ArrayQueue *q, __u32 k){

	q->queue[q->tail] = k;
	setBit(q->q_bitmap, k);
	q->tail = q->tail_next;
	q->tail++;
	q->tail_next++;

}


void enArrayQueueDelayed (struct ArrayQueue *q, __u32 k){

	q->queue[q->tail_next] = k;
	q->tail_next++;

}

void enArrayQueueDelayedWithBitmap (struct ArrayQueue *q, __u32 k){

	q->queue[q->tail_next] = k;
	setBit(q->q_bitmap_next, k);
	q->tail_next++;

}


void slideWindowArrayQueue (struct ArrayQueue *q){

		q->head = q->tail;
		q->tail = q->tail_next;
	
}

void slideWindowArrayQueueBitmap (struct ArrayQueue *q){

	q->head = q->tail;
	q->tail = q->tail_next;
	swapBitmaps(&q->q_bitmap, &q->q_bitmap_next);
	clearBitmap(q->q_bitmap_next);

}

__u32 deArrayQueue(struct ArrayQueue *q){

	__u32 k = q->queue[q->head];
	clearBit(q->q_bitmap,k);
	q->head = (q->head+1)%q->size;

	return k;

}


__u32 frontArrayQueue (struct ArrayQueue *q){

	__u32 k = q->queue[q->head];

	return k;

}

__u8 isEmptyArrayQueueCurr (struct ArrayQueue *q){

  if((q->tail > q->head))
  	return 0;
  else
  	return 1;

}

__u8 isEmptyArrayQueue (struct ArrayQueue *q){

  if(!isEmptyArrayQueueCurr(q) || !isEmptyArrayQueueNext(q))
  	return 0;
  else
  	return 1;

}

__u8 isEmptyArrayQueueNext (struct ArrayQueue *q){

  if((q->tail_next > q->head))
  	return 0;
  else
  	return 1;

}

__u8  isEnArrayQueued 	(struct ArrayQueue *q, __u32 k){


	return getBit(q->q_bitmap, k);

}

__u8  isEnArrayQueuedNext 	(struct ArrayQueue *q, __u32 k){


	return getBit(q->q_bitmap_next, k);

}

__u32 sizeArrayQueueCurr(struct ArrayQueue *q){

	return q->tail - q->head;

}

__u32 sizeArrayQueueNext(struct ArrayQueue *q){

	return q->tail_next - q->tail;
}


__u32 sizeArrayQueue(struct ArrayQueue *q){

	return q->tail_next - q->head;

}


/*

	Flush only works for single thread make sure is multi threaded safe

*/
#if CRITICAL_SECTION == LOCK
void flushArrayQueueToShared(struct ArrayQueue *local_q, struct ArrayQueue *shared_q, omp_lock_t flush_lock){
#else
void flushArrayQueueToShared(struct ArrayQueue *local_q, struct ArrayQueue *shared_q){
#endif

#if CRITICAL_SECTION == LOCK
	omp_set_lock(&flush_lock);
#elif CRITICAL_SECTION == CRITICAL
#pragma omp critical
{
#endif	
	// __u32 shared_q_tail_next = shared_q->tail_next; //every thread starts from its own starting point on
	// shared_q->tail_next += local_q->tail;
#if CRITICAL_SECTION == GCC	
	__u32 shared_q_tail_next = __sync_fetch_and_add(&shared_q->tail_next, local_q->tail);
#else
	__u32 shared_q_tail_next = shared_q->tail_next; //every thread starts from its own starting point on
	shared_q->tail_next += local_q->tail;
#endif
	__u32 local_q_size = local_q->tail - local_q->head;
	//reset the local queue state the ruse purpose is alloc/dealoc local queus create overhead.
	memcpy(&shared_q->queue[shared_q_tail_next],&local_q->queue[local_q->head],local_q_size*(sizeof(__u32)));
#if CRITICAL_SECTION == LOCK
    omp_unset_lock(&flush_lock);
#elif CRITICAL_SECTION == CRITICAL
}
#endif
	local_q->head = 0;
    local_q->tail = 0;
    local_q->tail_next = 0;
}



/*

you need to implement this if needed

*/
void enArrayQueueAtomic(struct ArrayQueue *q, __u32 k, omp_lock_t lock){
#if CRITICAL_SECTION == CRITICAL
#pragma omp critical
{
#elif CRITICAL_SECTION == LOCK
	omp_set_lock(&lock);
#endif
	// q->queue[q->tail] = k;
	// q->tail = (q->tail+1)%q->size;
	// q->tail_next = q->tail;
	q->queue[__sync_fetch_and_add(&q->tail, 1)] = k;
	q->tail_next = q->tail;
#if CRITICAL_SECTION == CRITICAL
}
#elif CRITICAL_SECTION == LOCK
	omp_unset_lock(&lock);
#endif
}



void enArrayQueue (struct ArrayQueue *q, __u32 k){

	q->queue[q->tail] = k;
	q->tail = (q->tail+1)%q->size;
	q->tail_next = q->tail;

}
