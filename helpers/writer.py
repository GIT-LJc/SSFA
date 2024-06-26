def write_acc(args, test_acc, cortest_acc, path):
    path = path + '/Acc.txt'
    with open(path, 'a') as f:
        f.write( 'model: ' + args.resume + '  epoch: ' + str(args.start_epoch) + '\n')
        f.write('unlabeled dataset----   corruption: '+ args.corruption + ' level: ' + str(args.corruption_level) + ' ratio: ' + str(args.ratio) + '\n')
        f.write("test acc: {:.2f}".format(test_acc) + '\n')
        f.write("corruption test acc: {:.2f}".format(cortest_acc) + '\n')
        f.write("\n")
    f.close()
    print("write done!")

    