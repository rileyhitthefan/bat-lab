SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;

SET @OLD_SQL_MODE = @@SQL_MODE;
SET SQL_MODE = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- Schema
CREATE SCHEMA IF NOT EXISTS `batlab_schema` DEFAULT CHARACTER SET utf8;
USE `batlab_schema`;

-- Locations table
DROP TABLE IF EXISTS Call_Library;
DROP TABLE IF EXISTS Location_Bats;
DROP TABLE IF EXISTS Bats;
DROP TABLE IF EXISTS Locations;

CREATE TABLE Locations (
  area_name VARCHAR(45) NOT NULL,
  latitude DECIMAL(6,3) NOT NULL,
  longitude DECIMAL(6,3) NOT NULL,
  PRIMARY KEY (area_name),
  UNIQUE INDEX area_name_UNIQUE (area_name ASC)
) ENGINE=InnoDB;
-- Bats table

CREATE TABLE IF NOT EXISTS Bats (
  `abbreviation` VARCHAR(45) NOT NULL,
  `latin_name` VARCHAR(45) NULL,
  `common_name` VARCHAR(45) NULL,
  PRIMARY KEY (`abbreviation`),
  UNIQUE INDEX `latin_name_UNIQUE` (`latin_name` ASC),
  UNIQUE INDEX `abbreviation_UNIQUE` (`abbreviation` ASC),
  UNIQUE INDEX `common_name_UNIQUE` (`common_name` ASC)
) ENGINE=InnoDB;

-- Location_Bats table
CREATE TABLE IF NOT EXISTS Location_Bats (
  `id` INT NOT NULL AUTO_INCREMENT,
  `bat_abbreviation` VARCHAR(45) NOT NULL,
  `location_name` VARCHAR(45) NOT NULL,
  `lowest_frequency` INT NULL,
  `highest_frequency` INT NULL,
  `shape` BLOB NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `bat_location_unique` (`bat_abbreviation`, `location_name`),
  CONSTRAINT `fk_bat` FOREIGN KEY (`bat_abbreviation`)
    REFERENCES `Bats` (`abbreviation`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `fk_location` FOREIGN KEY (`location_name`)
    REFERENCES `Locations` (`area_name`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB;

-- Call_Library table

CREATE TABLE IF NOT EXISTS Call_Library (
  `idTraining` INT NOT NULL AUTO_INCREMENT,
  `files` LONGBLOB NOT NULL,
  `bat_abbreviation` VARCHAR(45) NOT NULL,
  `location_name` VARCHAR(45) NOT NULL,
  `file_hash` CHAR(64) NOT NULL UNIQUE,
  PRIMARY KEY (`idTraining`),
  UNIQUE INDEX `idTraining_UNIQUE` (`idTraining` ASC),
  UNIQUE INDEX `file_hash_UNIQUE` (`file_hash` ASC),
  CONSTRAINT `fk_bat_location` FOREIGN KEY (`bat_abbreviation`, `location_name`)
    REFERENCES `Location_Bats` (`bat_abbreviation`, `location_name`)
    ON DELETE CASCADE ON UPDATE cascade
) ENGINE=InnoDB;

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
