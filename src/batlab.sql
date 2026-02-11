SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE = @@SQL_MODE;
SET SQL_MODE = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- Schema
CREATE SCHEMA IF NOT EXISTS `batlab_schema` DEFAULT CHARACTER SET utf8;
USE `batlab_schema`;

-- Drop tables in correct order (respecting foreign keys)
DROP TABLE IF EXISTS `Call_Library`;
DROP TABLE IF EXISTS `Location_Bats`;
DROP TABLE IF EXISTS `Bats`;
DROP TABLE IF EXISTS `Locations`;

-- Locations table
CREATE TABLE `Locations` (
  `Detector_name` VARCHAR(45) NOT NULL,
  `latitude` DECIMAL(10,8) NOT NULL,
  `longitude` DECIMAL(11,8) NOT NULL,
  PRIMARY KEY (`Detector_name`),
  UNIQUE INDEX `Detector_name_UNIQUE` (`Detector_name` ASC)
) ENGINE=InnoDB;

-- Bats table
CREATE TABLE IF NOT EXISTS `Bats` (
  `abbreviation` VARCHAR(45) NOT NULL,
  `latin_name` VARCHAR(45) NULL,
  `common_name` VARCHAR(45) NULL,
  PRIMARY KEY (`abbreviation`),
  UNIQUE INDEX `latin_name_UNIQUE` (`latin_name` ASC),
  UNIQUE INDEX `abbreviation_UNIQUE` (`abbreviation` ASC),
  UNIQUE INDEX `common_name_UNIQUE` (`common_name` ASC)
) ENGINE=InnoDB;

-- Call_Library table
CREATE TABLE IF NOT EXISTS `Call_Library` (
  `idTraining` INT NOT NULL AUTO_INCREMENT,
  `files` LONGBLOB NOT NULL,
  `latin_name` VARCHAR(45) NOT NULL,
  `Detector_name` VARCHAR(45) NOT NULL,
  `file_hash` CHAR(64) AS (SHA2(`filename`, 256)) STORED,NOT NULL UNIQUE ,
  PRIMARY KEY (`idTraining`),
  UNIQUE INDEX `idTraining_UNIQUE` (`idTraining` ASC),
  UNIQUE INDEX `file_hash_UNIQUE` (`file_hash` ASC),
  CONSTRAINT `fk_bats` FOREIGN KEY (`latin_name`)
    REFERENCES `Bats` (`latin_name`)
    ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_location` FOREIGN KEY (`Detector_name`)
    REFERENCES `Locations` (`Detector_name`)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
