class CompatibleNet(nn.Module):
    def __init__(self):
        super(CompatibleNet, self).__init__()
        self.conv1 = nn.Conv2d(150, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.up = nn.ConvTranspose2d(64, 16, 2, stride=2)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        pool1 = self.max_pool(conv1)
        conv2 = F.relu(self.conv2(pool1))
        drop2 = self.dropout(conv2)

        up3 = F.relu(self.up(drop2))
        merge3 = torch.cat([conv1, up3], dim=1)
        conv3 = F.relu(self.conv3(merge3))
        conv4 = F.softmax(self.conv4(conv3), dim=1)

        return conv4