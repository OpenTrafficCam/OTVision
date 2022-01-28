import unittest
import os
from pathlib import Path
import shutil

from OTVision.helpers.files import get_files


class FileTests(unittest.TestCase):
    def setUp(self):
        self.testDirPath = str(Path(__file__).parents[1] / "resources" / "test_dir")
        self.fileNames = ["readme.txt", "cities.json", "configurations.xml"]
        createTestDir(self.testDirPath)
        createTestFiles(self.testDirPath, self.fileNames)

    def test_get_files_correctParam_returnsCorrectList(self):
        jsonFilePath = str(os.path.join(self.testDirPath, "cities.json"))
        xmlFilePath = str(os.path.join(self.testDirPath, "configurations.xml"))

        files = get_files(self.testDirPath, [".json", ".xml"])

        self.assertTrue(jsonFilePath in files)
        self.assertTrue(xmlFilePath in files)

    def test_get_files_noFilenamesAs2ndParam_ReturnEmptyList(self):
        files = get_files(self.testDirPath, [])
        self.assertEqual([], files)

    def test_get_files_invalidTypeAs1stParam_RaiseTypeError(self):
        notStringPath = Path(__file__)
        self.assertRaises(TypeError, get_files, notStringPath, [".json"])

    def tearDown(self):
        shutil.rmtree(self.testDirPath)


def createTestDir(pathToDir):
    os.makedirs(name=pathToDir, exist_ok=True)


def createTestFiles(pathToDir, fileNames):
    for name in fileNames:
        filePath = os.path.join(pathToDir, name)
        file = open(filePath, "w+")
        file.close()


if __name__ == "__main__":
    unittest.main()
