.pos 0x1000
start: #setup
  ld $stacktop, r6
  gpc $6, r6
  j main
  halt

main:
  deca r5 # allocate 4 bytes
  st r6, (r5) # store ra on stack
  ld $-128, r0 # allocate 128 bytes on stack
  add r0, r5 # cont.
  deca r5 # allocate 4 bytes on stack

  ld $str1, r0 # r0 = str1
  st r0, (r5) # load str1 ptr to the stack
  gpc $6, r6 # setup ra
  j print # jump to print
  inca r5 # dealloc 4 bytes (str1 ptr)
  ld $0, r0 # load 0 to r0
  mov r5, r1 # r1 = stackptr
  ld $256, r2 # load 256 to r2
  sys $0 # read data (stdin, buffer=stackptr, size=256)
  mov r0, r4 # r4 = read bytes
  deca r5 # allocate 4 bytes on stack
  ld $str2, r0 # load str2 ptr to r0
  st r0, (r5) # push str2 to stack
  gpc $6, r6 # call print on str2
  j print # cont.
  inca r5 # dealloc str2 ptr
  ld $1, r0 # r0 = 1
  mov r5, r1 # r1 = stackptr
  mov r4, r2 # r2 = # bytes read
  sys $1 # write user input string bytes to stdout
  ld $128, r0 # load 128 to r0
  add r0, r5 # deallocate 128 bytes off stack
  ld (r5), r6 # return
  inca r5
  j (r6)

print:
  ld (r5), r0 # pull top value off stack (str ptr)
  ld 4(r0), r1 # r1 = beginning of str buffer
  ld (r0), r2 # r2 = length of str buffer
  ld $1, r0 # r0 = stdout
  sys $1 # print string to stdout
  j (r6)

.pos 0x1800
proof:
  deca r5 # allocate stack memory
  ld $str3, r0 # load str3 ptr to r0
  st r0, (r5) # store str3 part onto stack
  gpc $6, r6 # call print
  j print
  halt

.pos 0x2000
str1:
  .long 30
  .long _str1
_str1:
  .long 0x57656c63
  .long 0x6f6d6521
  .long 0x20506c65
  .long 0x61736520
  .long 0x656e7465
  .long 0x72206120
  .long 0x6e616d65
  .long 0x3a0a0000

str2:
  .long 11
  .long _str2
_str2:
  .long 0x476f6f64
  .long 0x206c7563
  .long 0x6b2c2000

str3:
  .long 43
  .long _str3
_str3:
  .long 0x54686520
  .long 0x73656372
  .long 0x65742070
  .long 0x68726173
  .long 0x65206973
  .long 0x20227371
  .long 0x7565616d
  .long 0x69736820
  .long 0x6f737369
  .long 0x66726167
  .long 0x65220a00

.pos 0x4000
stack:
  .long 0
 .long 0
stacktop:
  .long 0


